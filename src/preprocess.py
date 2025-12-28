import os, glob
import numpy as np
from PIL import Image

DATA_DIR = "data/raw/manual"
OUT_X = "outputs/X.npy"
OUT_Y = "outputs/y.npy"
OUT_NAMES = "outputs/names.txt"

IMG_SIZE = (224, 224)  # tamaño estándar

def label_from_name(fname: str) -> int:
    base = os.path.basename(fname).lower()
    if base.startswith("im-"):
        return 0  # normal
    if "bacteria" in base or "virus" in base:
        return 1  # neumonía
    return -1    # desconocido

def main():
    paths = sorted([p for p in glob.glob(os.path.join(DATA_DIR, "*"))
                    if p.lower().endswith((".jpg",".jpeg",".png"))])

    if not paths:
        raise SystemExit(f"No hay imágenes en {DATA_DIR}")

    X_list, y_list, names = [], [], []

    for p in paths:
        y = label_from_name(p)
        if y == -1:
            print(f"⚠️  Etiqueta desconocida, salto: {p}")
            continue

        img = Image.open(p).convert("L")          # escala de grises
        img = img.resize(IMG_SIZE)               # redimensionar
        arr = np.array(img, dtype=np.float32)    # (224,224)
        arr = arr / 255.0                        # normalizar 0..1

        X_list.append(arr)
        y_list.append(y)
        names.append(os.path.basename(p))

    X = np.stack(X_list, axis=0)                 # (N,224,224)
    y = np.array(y_list, dtype=np.int64)         # (N,)

    os.makedirs("outputs", exist_ok=True)
    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    with open(OUT_NAMES, "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    print("✅ Dataset creado")
    print(" - X:", X.shape, X.dtype, "min/max:", float(X.min()), float(X.max()))
    print(" - y:", y.shape, y.dtype, "conteo [normal=0, neumonía=1]:",
          int((y==0).sum()), int((y==1).sum()))
    print(" - Guardado en outputs/: X.npy, y.npy, names.txt")

if __name__ == "__main__":
    main()
