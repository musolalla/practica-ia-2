import os, glob, csv
import numpy as np
import joblib
import cv2
from PIL import Image

MODEL_PATH = "outputs/models/tamper_lr.joblib"
IN_DIR = "data/raw/manual"
IMG_SIZE = (224, 224)

def preprocess_gray(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (224,224) float 0..1

def segment_lungs_mask(img_u8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x = clahe.apply(img_u8)
    x = cv2.GaussianBlur(x, (5,5), 0)
    _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(th)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    areas.sort(key=lambda t: t[1], reverse=True)
    keep = [areas[0][0]] + ([areas[1][0]] if len(areas) > 1 else [])
    mask = np.zeros_like(th)
    for lab in keep:
        mask[labels == lab] = 255
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = (mask > 60).astype(np.uint8) * 255
    return mask

def split_left_right(mask):
    h, w = mask.shape
    mid = w // 2
    left = np.zeros_like(mask); right = np.zeros_like(mask)
    left[:, :mid] = mask[:, :mid]
    right[:, mid:] = mask[:, mid:]
    return left, right

def prob_for_region(clf, img01, mask_u8):
    # aplicar máscara: fuera de región = 0
    m = (mask_u8.astype(np.float32) / 255.0)
    region = img01 * m
    X = region.reshape(1, -1)
    return float(clf.predict_proba(X)[0, 1])

def status(prob, thr=0.5):
    return "MANIPULADA" if prob >= thr else "Original"

def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No encuentro el modelo: {MODEL_PATH}. Ejecuta train_tamper_detector.py primero.")

    clf = joblib.load(MODEL_PATH)

    paths = sorted([p for p in glob.glob(os.path.join(IN_DIR, "*"))
                    if p.lower().endswith((".jpg",".jpeg",".png"))])

    out_csv = "outputs/report_tamper_by_region.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Imagen ID", "Región Analizada", "Prob. Manipulación", "Estado"])

        for p in paths:
            img01 = preprocess_gray(p)
            img_u8 = (img01 * 255).astype(np.uint8)

            mask = segment_lungs_mask(img_u8)
            left, right = split_left_right(mask)

            p_left = prob_for_region(clf, img01, left)
            p_right = prob_for_region(clf, img01, right)

            base = os.path.basename(p)
            w.writerow([base, "Pulmón Izq.", f"{p_left*100:.1f}%", status(p_left)])
            w.writerow([base, "Pulmón Der.", f"{p_right*100:.1f}%", status(p_right)])

    print("✅ CSV por región creado:", out_csv)
    print("Vista previa:")
    with open(out_csv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 6:
                break

if __name__ == "__main__":
    main()
