import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.load("outputs/X.npy")   # (N,224,224)
y = np.load("outputs/y.npy")   # (N,)

# Aplanar imágenes: (N,224*224)
Xf = X.reshape(X.shape[0], -1)

# Con tan pocas imágenes, usamos split simple y estratificado
X_train, X_test, y_train, y_test = train_test_split(
    Xf, y, test_size=0.25, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("✅ Baseline entrenado")
print("Test size:", len(y_test))
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, digits=3))
