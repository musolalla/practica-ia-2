import os, glob, csv
import numpy as np
import joblib
from PIL import Image

MODEL_PATH = "outputs/models/tamper_lr.joblib"
IN_DIR = "data/raw/manual"
IMG_SIZE = (224, 224)

def preprocess(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

def status_from_prob(p, thr=0.5):
    return "MANIPULADA" if p >= thr else "Original"

def main():
    clf = joblib.load(MODEL_PATH)

    paths = sorted([p for p in glob.glob(os.path.join(IN_DIR, "*"))
                    if p.lower().endswith((".jpg",".jpeg",".png"))])

    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/report_tamper.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Imagen ID", "Región Analizada", "Prob. Manipulación", "Estado"])
        for p in paths:
            X = preprocess(p)
            prob = float(clf.predict_proba(X)[0, 1])  # prob clase 1 = manipulada
            w.writerow([os.path.basename(p), "Global", f"{prob*100:.1f}%", status_from_prob(prob)])

    print(f"✅ Reporte generado: {out_csv}")
    print("Muestra (primeras filas):")
    with open(out_csv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 5:
                break

if __name__ == "__main__":
    main()
