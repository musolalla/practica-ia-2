import os, glob, random
import numpy as np
from PIL import Image, ImageFilter

IN_DIR = "data/raw/manual"
OUT_DIR = "outputs/tamper"
IMG_SIZE = (224, 224)

random.seed(42)
np.random.seed(42)

def load_gray(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def add_blob(arr):
    """Añade una 'mancha' circular suave."""
    h, w = arr.shape
    y0 = random.randint(h//4, 3*h//4)
    x0 = random.randint(w//4, 3*w//4)
    r = random.randint(12, 26)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - y0)**2 + (xx - x0)**2 <= r*r
    blob = np.zeros_like(arr)
    blob[mask] = random.uniform(0.3, 0.7)
    # suavizar
    pil = Image.fromarray(np.uint8(np.clip(blob*255,0,255)))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=6))
    blob = np.array(pil, dtype=np.float32)/255.0
    out = np.clip(arr + blob*0.6, 0, 1)
    return out

def erase_patch(arr):
    """Borra una zona: parche con blur (simula 'borrado')."""
    h, w = arr.shape
    ph = random.randint(30, 60)
    pw = random.randint(30, 60)
    y0 = random.randint(0, h - ph)
    x0 = random.randint(0, w - pw)
    patch = arr[y0:y0+ph, x0:x0+pw].copy()

    pil = Image.fromarray(np.uint8(np.clip(patch*255,0,255)))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=8))
    blur = np.array(pil, dtype=np.float32)/255.0

    out = arr.copy()
    out[y0:y0+ph, x0:x0+pw] = blur
    return out

def add_noise(arr):
    """Añade ruido leve."""
    sigma = random.uniform(0.02, 0.06)
    noise = np.random.normal(0, sigma, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0, 1)

def main():
    paths = sorted([p for p in glob.glob(os.path.join(IN_DIR, "*"))
                    if p.lower().endswith((".jpg",".jpeg",".png"))])
    if not paths:
        raise SystemExit(f"No hay imágenes en {IN_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, names = [], [], []

    for p in paths:
        base = os.path.basename(p)
        orig = load_gray(p)
        X.append(orig); y.append(0); names.append(f"orig__{base}")

        # 3 variantes manipuladas
        variants = [
            ("blob", add_blob(orig)),
            ("erase", erase_patch(orig)),
            ("noise", add_noise(orig)),
        ]
        for tag, arr in variants:
            X.append(arr); y.append(1); names.append(f"{tag}__{base}")

    X = np.stack(X, axis=0).astype(np.float32)  # (N,224,224)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(OUT_DIR, "X_tamper.npy"), X)
    np.save(os.path.join(OUT_DIR, "y_tamper.npy"), y)
    with open(os.path.join(OUT_DIR, "names_tamper.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names))

    print("✅ Dataset manipulación creado")
    print(" - X:", X.shape, "y:", y.shape)
    print(" - originales:", int((y==0).sum()), "manipuladas:", int((y==1).sum()))
    print(" - Guardado en outputs/tamper/")
    print("   * X_tamper.npy, y_tamper.npy, names_tamper.txt")

if __name__ == "__main__":
    main()
