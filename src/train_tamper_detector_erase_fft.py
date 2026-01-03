import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

IN_DIR = "outputs/tamper_erase"
MODEL_OUT = "outputs/models/tamper_erase_fft_lr.joblib"

def fft_features(img: np.ndarray, out_size: int = 32) -> np.ndarray:
    """
    img: (H,W) float32 in [0,1]
    Returns: (out_size*out_size,) feature vector from log-magnitude FFT (center-cropped + pooled).
    """
    # FFT magnitude (log)
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift))

    # Normalize per-image to reduce brightness/contrast influence
    mag = (mag - mag.mean()) / (mag.std() + 1e-8)

    H, W = mag.shape
    # Simple block pooling to out_size x out_size
    bh = H // out_size
    bw = W // out_size
    mag = mag[:bh*out_size, :bw*out_size]  # crop to divisible
    pooled = mag.reshape(out_size, bh, out_size, bw).mean(axis=(1,3))

    return pooled.reshape(-1)

# Load
X = np.load(os.path.join(IN_DIR, "X_tamper.npy"))  # (N,224,224)
y = np.load(os.path.join(IN_DIR, "y_tamper.npy"))  # (N,)

# Build feature matrix
X_feat = np.stack([fft_features(X[i]) for i in range(X.shape[0])], axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=5000, class_weight="balanced")
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("✅ Detector FFT entrenado (ERASE)")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

os.makedirs("outputs/models", exist_ok=True)
joblib.dump(clf, MODEL_OUT)
print(f"✅ Modelo guardado en {MODEL_OUT}")
