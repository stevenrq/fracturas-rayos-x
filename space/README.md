---
title: Detección de Fracturas Óseas
emoji: 🦴
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: mit
short_description: Demo académica de detección de fracturas en radiografías
---

<!--
  IMPORTANTE antes de publicar:
  1. Añade arriba la línea  `sdk_version: X.Y.Z`  con tu versión local de Gradio
     (mírala con `gradio --version`). Si la omites, HF usa la última.
  2. Sube el modelo `best_mobilenet.keras` a un Model repo de HF
     (p. ej. stevenrq8/fracturas-modelo) o ponlo junto a app.py con Git LFS.
  3. Si cambias el repo/archivo/umbral, configúralo en
     Settings → Variables: MODEL_REPO, MODEL_FILE, UMBRAL.
-->

# 🦴 Detección de fracturas óseas en radiografías

Demo del proyecto final de **Aprendizaje Computacional** (Ingeniería de Sistemas,
Universidad de Córdoba). Sube una radiografía y el modelo (MobileNetV2 con
*transfer learning*) estima la **probabilidad de fractura**, muestra la etiqueta y
la confianza, y resalta con **Grad-CAM** la zona que observó.

## Cómo usar

1. Sube una imagen de radiografía (`.jpg`, `.png`).
2. Lee la probabilidad de fractura, la etiqueta y el mapa de calor.

El umbral de decisión está ajustado para **priorizar la sensibilidad** (detectar
fracturas), por lo que puede generar falsos positivos a propósito.

## ⚠️ Aviso médico

Esta aplicación es una **demostración académica y educativa**. **No** es un
dispositivo médico ni está validada clínicamente. Sus predicciones son
orientativas, pueden contener errores y **no sustituyen el diagnóstico de un
profesional de la salud**. No debe usarse para decisiones clínicas reales.

## Enlaces

- Código y documentación: https://github.com/stevenrq8/fracturas-rayos-x
- Dataset: *Bone Fracture Multi-Region X-ray Data* (Kaggle, `bmadushanirodrigo`, licencia ODC-By v1.0).
