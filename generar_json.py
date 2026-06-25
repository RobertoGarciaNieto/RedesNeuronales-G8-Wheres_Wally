import os
import json
from PIL import Image

CLASS_NAMES = {0: "odlaw", 1: "waldo", 2: "wilma", 3: "wizard", 4: "woof"}

NIVELES = [
    ("facil",        "facil.jpg",         "Fácil"),
    ("medio",        "medio.jpg",         "Medio"),
    ("dificil",      "dificil.jpg",       "Difícil"),
    ("superDificil", "superDificil.webp", "Super Dificil"),
]

ground_truth = {}

if not os.path.exists("posters_fijos"):
    print("Error: Crea la carpeta 'posters_fijos' y poné tus archivos ahí.")
    exit()

for nivel_key, img_file, nivel_label in NIVELES:
    img_path = f"posters_fijos/{img_file}"
    txt_path = f"posters_fijos/{nivel_key}.txt"

    if not os.path.exists(img_path):
        print(f"⚠️  No se encontró imagen: {img_path} — saltando.")
        continue
    if not os.path.exists(txt_path):
        print(f"⚠️  No se encontró anotación: {txt_path} — saltando.")
        continue

    img = Image.open(img_path)
    w_img, h_img = img.size
    anotaciones = []

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                x1 = int((cx - w / 2) * w_img)
                y1 = int((cy - h / 2) * h_img)
                x2 = int((cx + w / 2) * w_img)
                y2 = int((cy + h / 2) * h_img)
                anotaciones.append({
                    "personaje": CLASS_NAMES.get(cls_id, "desconocido"),
                    "coords": [x1, y1, x2, y2]
                })

    ground_truth[nivel_label] = anotaciones
    print(f"✅  {nivel_label}: {len(anotaciones)} anotaciones cargadas.")

with open("posters_fijos/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=4, ensure_ascii=False)

print("\n¡ground_truth.json generado con éxito en posters_fijos/!")