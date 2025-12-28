import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = np.load("outputs/tamper/X_tamper.npy")
y = np.load("outputs/tamper/y_tamper.npy")
Xf = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(
    Xf, y, test_size=0.25, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("✅ Detector entrenado")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

os.makedirs("outputs/models", exist_ok=True)
joblib.dump(clf, "outputs/models/tamper_lr.joblib")
print("✅ Modelo guardado en outputs/models/tamper_lr.joblib")
