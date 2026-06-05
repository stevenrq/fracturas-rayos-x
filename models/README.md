---
license: mit
language:
  - es
base_model:
  - google/mobilenet_v2_1.0_224
pipeline_tag: image-classification
tags:
  - medical
  - radiology
  - fracture-detection
  - keras
  - tensorflow
  - transfer-learning
  - grad-cam
datasets:
  - bmadushanirodrigo/bone-fracture-multi-region-x-ray-data
metrics:
  - accuracy
  - f1
  - auc
model-index:
  - name: best_mobilenet
    results:
      - task:
          type: image-classification
          name: Bone Fracture Detection
        dataset:
          name: Bone Fracture Multi-Region X-ray Data (deduplicated)
          type: bmadushanirodrigo/bone-fracture-multi-region-x-ray-data
          split: test
        metrics:
          - type: accuracy
            value: 0.815
          - type: recall
            value: 0.953
            name: Recall (fractura, umbral 0.310)
          - type: f1
            value: 0.782
          - type: auc
            value: 0.910
---

# 🦴 Detección de fracturas óseas — MobileNetV2

Modelo de clasificación binaria (fractura / normal) sobre radiografías óseas.
Desarrollado como proyecto final de **Aprendizaje Computacional** (Ingeniería de
Sistemas, Universidad de Córdoba).

Arquitectura: **MobileNetV2** con *transfer learning* desde ImageNet + cabeza de
clasificación personalizada. Umbral de decisión ajustado a **0.310** para
priorizar la sensibilidad clínica (recall ≥ 0.95).

> ⚠️ **Demostración académica.** No es un dispositivo médico ni está validado
> clínicamente. No debe usarse para decisiones clínicas reales.

## Demo interactiva

Prueba el modelo en el Space de Hugging Face:
👉 [stevenrq8/fracturas-rayos-x](https://huggingface.co/spaces/stevenrq8/fracturas-rayos-x)

## Uso

```python
from huggingface_hub import hf_hub_download
from tensorflow import keras
import numpy as np
from PIL import Image

model = keras.models.load_model(
    hf_hub_download("stevenrq8/fracturas-modelo", "best_mobilenet.keras")
)

UMBRAL = 0.310
IMG_SIZE = (224, 224)

img = Image.open("radiografia.jpg").convert("RGB").resize(IMG_SIZE)
arr = np.asarray(img, dtype="float32")[None, ...]
p = float(model.predict(arr).ravel()[0])
etiqueta = "FRACTURA" if p >= UMBRAL else "NORMAL"
print(f"{etiqueta}  —  P(fractura) = {p:.1%}")
```

## Datos de entrenamiento

Dataset base: *Bone Fracture Multi-Region X-ray Data*
(Kaggle, `bmadushanirodrigo`, licencia ODC-By v1.0).

El split oficial tenía 63 % de duplicados y 873 grupos con fuga entre
train/val/test. Se deduplicó y re-dividió antes del entrenamiento:

| Split | Normal | Fractura | Total |
|-------|-------:|---------:|------:|
| Train |  1 611 |    1 127 | 2 738 |
| Val   |    343 |      244 |   587 |
| Test  |    328 |      235 |   563 |

Desbalance leve (59 % normal / 41 % fractura) corregido con `class_weight`.

## Resultados (test set, umbral 0.310)

| Métrica | Valor |
|---------|------:|
| Accuracy | 0.815 |
| Recall (fractura) | **0.953** |
| Especificidad | 0.649 |
| Precision | 0.661 |
| F1 | 0.781 |
| AUC-ROC | **0.910** |
| Tiempo inferencia (CPU) | 0.070 s |
| Tamaño del modelo | 9.3 MB |

### Comparativa de modelos (umbral 0.5)

| Modelo | Accuracy | Recall | AUC | Tamaño |
|--------|:--------:|:------:|:---:|-------:|
| CNN desde cero | 0.673 | 0.243 | 0.820 | 1.2 MB |
| **MobileNetV2** ✅ | **0.815** | **0.796** | 0.910 | 9.3 MB |
| EfficientNetB0 | 0.798 | 0.681 | **0.919** | 43.4 MB |

Se eligió MobileNetV2 por su mejor equilibrio entre recall, AUC y tamaño
(viable en hardware CPU gratuito).

## Entrenamiento

- Framework: TensorFlow / Keras
- Hardware: Google Colab (GPU T4)
- Épocas: EarlyStopping (paciencia 5) + ReduceLROnPlateau
- Regularización: Dropout 0.3, data augmentation, class_weight
- Fine-tuning: descongelación de las últimas capas de MobileNetV2

## Limitaciones

- Dataset pequeño (3 888 imágenes únicas) — sobreajuste moderado (train AUC ~0.96 vs val ~0.84)
- Mezcla de regiones anatómicas sin etiqueta explícita (muñeca, hombro, rodilla…)
- Sin validación clínica por radiólogos
- Una sola vista por caso, sin datos demográficos del paciente

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `best_mobilenet.keras` | Modelo final elegido para despliegue |

## Licencia

MIT — ver [LICENSE](https://github.com/stevenrq8/fracturas-rayos-x/blob/main/LICENSE).
