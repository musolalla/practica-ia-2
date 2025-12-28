import os, glob
import numpy as np
import cv2
from PIL import Image

IN_DIR = "data/raw/manual"
OUT_DIR = "outputs/seg"
IMG_SIZE = (224, 224)

def segment_lungs_mask(img_u8: np.ndarray) -> np.ndarray:
    """
    Segmentación simple aproximada de pulmones en RX:
    - realce contraste
    - umbral Otsu (invertido)
    - morfología (open/close)
    - quedarse con 2 componentes más grandes (pulmones)
    """
    # CLAHE para contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x = clahe.apply(img_u8)

    # Suavizado
    x = cv2.GaussianBlur(x, (5,5), 0)

    # Otsu invertido: en RX, pulmones suelen ser más oscuros
    _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Quitar ruido y rellenar huecos
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    # Componentes conectados: quedarnos con 2 mayores (pulmones)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(th)

    # stats: [label, x, y, w, h, area] ; label 0 = fondo
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    areas.sort(key=lambda t: t[1], reverse=True)
    keep = [areas[0][0]]
    if len(areas) > 1:
        keep.append(areas[1][0])

    mask = np.zeros_like(th)
    for lab in keep:
        mask[labels == lab] = 255

    # Un pequeño suavizado del borde
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = (mask > 60).astype(np.uint8) * 255
    return mask

def split_left_right(mask: np.ndarray):
    """Divide máscara en izquierda/derecha por la línea media (x=112)."""
    h, w = mask.shape
    mid = w // 2
    left = np.zeros_like(mask); right = np.zeros_like(mask)
    left[:, :mid] = mask[:, :mid]
    right[:, mid:] = mask[:, mid:]
    return left, right

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted([p for p in glob.glob(os.path.join(IN_DIR, "*"))
                    if p.lower().endswith((".jpg",".jpeg",".png"))])

    for p in paths:
        base = os.path.basename(p)
        img = Image.open(p).convert("L").resize(IMG_SIZE)
        img_u8 = np.array(img, dtype=np.uint8)

        mask = segment_lungs_mask(img_u8)
        left, right = split_left_right(mask)

        # Guardar png para inspección visual
        cv2.imwrite(os.path.join(OUT_DIR, base.replace(".jpeg","").replace(".jpg","").replace(".png","") + "__mask.png"), mask)
        cv2.imwrite(os.path.join(OUT_DIR, base.replace(".jpeg","").replace(".jpg","").replace(".png","") + "__left.png"), left)
        cv2.imwrite(os.path.join(OUT_DIR, base.replace(".jpeg","").replace(".jpg","").replace(".png","") + "__right.png"), right)

    print(f"✅ Máscaras generadas en {OUT_DIR} (mask/left/right por imagen)")

if __name__ == "__main__":
    main()
