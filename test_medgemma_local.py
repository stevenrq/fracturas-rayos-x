"""
Test local de MedGemma (google/medgemma-4b-it) con cuantizacion 4-bit.
Requiere: GPU NVIDIA con >=4GB VRAM, CUDA, PyTorch con soporte CUDA,
          transformers>=5.0, bitsandbytes, huggingface_hub, Pillow.
Probado en: RTX 3050 Laptop (4.3 GB VRAM), torch 2.6+cu124, transformers 5.10.2

Resultado esperado: descripcion textual en espanol de la radiografia de prueba
generada por MedGemma directamente en la GPU local.
"""
import os
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Pon tu token aqui o en variable de entorno HF_TOKEN
login(token=HF_TOKEN)

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import glob

print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

MODEL_ID = "google/medgemma-4b-it"

# Cuantizacion 4-bit: reduce el modelo de ~8GB a ~2GB, cabe en 4GB VRAM
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

print("\nCargando processor y modelo...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()
print("Modelo listo.")

# Cargar imagen de ejemplo del proyecto
imagenes = (
    glob.glob("ejemplos/*.jpg") + glob.glob("ejemplos/*.png") +
    glob.glob("app/ejemplos/*.jpg") + glob.glob("app/ejemplos/*.png")
)
if imagenes:
    image = Image.open(imagenes[0]).convert("RGB")
    print("Imagen de prueba:", imagenes[0])
else:
    import numpy as np
    arr = (np.random.rand(224, 224) * 200 + 55).astype("uint8")
    image = Image.fromarray(arr, mode="L").convert("RGB")
    print("Usando imagen sintetica (no se encontraron imagenes de ejemplo)")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                "Eres un asistente educativo de radiologia. "
                "Describe brevemente esta radiografia en espanol en 3-4 oraciones. "
                "Menciona que estructuras son visibles y si hay algo destacable."
            )},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
inputs.pop("token_type_ids", None)

print("Generando respuesta con MedGemma...")
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
respuesta = processor.decode(new_tokens, skip_special_tokens=True)

print("\n=== RESPUESTA DE MEDGEMMA ===")
print(respuesta)
print("=============================")
