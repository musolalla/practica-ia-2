import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

DATA_DIR = "outputs/tamper_noise"
MODEL_PATH = "outputs/models/tamper_torch_base.pt"
OUT_CSV = "outputs/report_adv_fgsm.csv"

BATCH_SIZE = 32
EPS = float(os.environ.get("EPS", "0.01"))  # epsilon por entorno
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def fgsm_attack(model, x, y, eps):
    x = x.clone().detach().to(DEVICE)
    x.requires_grad = True
    y = y.to(DEVICE)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad()
    loss.backward()

    grad_sign = x.grad.detach().sign()
    x_adv = x + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv

# Load data
X = np.load(os.path.join(DATA_DIR, "X_tamper.npy"))
y = np.load(os.path.join(DATA_DIR, "y_tamper.npy"))

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Load model
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluate clean + FGSM
clean_preds, adv_preds, trues = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        clean_preds.extend(out.argmax(1).cpu().numpy())

# FGSM needs grads -> no torch.no_grad
for xb, yb in test_loader:
    xb_adv = fgsm_attack(model, xb, yb, EPS)
    out_adv = model(xb_adv.to(DEVICE))
    adv_preds.extend(out_adv.argmax(1).detach().cpu().numpy())
    trues.extend(yb.numpy())

clean_acc = accuracy_score(trues, clean_preds)
adv_acc = accuracy_score(trues, adv_preds)

print("✅ Evaluación FGSM")
print("Epsilon:", EPS)
print("Clean accuracy:", clean_acc)
print("Clean confusion:\n", confusion_matrix(trues, clean_preds))
print("FGSM accuracy:", adv_acc)
print("FGSM confusion:\n", confusion_matrix(trues, adv_preds))

# Save CSV summary
os.makedirs("outputs", exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epsilon", "clean_accuracy", "fgsm_accuracy"])
    w.writerow([EPS, clean_acc, adv_acc])

print(f"✅ Guardado resumen en {OUT_CSV}")
