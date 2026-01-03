import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = "outputs/tamper_noise"
MODEL_OUT = "outputs/models/tamper_torch_base.pt"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load(os.path.join(DATA_DIR, "X_tamper.npy"))
y = np.load(os.path.join(DATA_DIR, "y_tamper.npy"))

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,224,224)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

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

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    last_loss = None
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {last_loss:.4f}")

model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(yb.numpy())

print("✅ Detector PyTorch entrenado (SIMPLE)")
print("Accuracy:", accuracy_score(trues, preds))
print("Confusion matrix:\\n", confusion_matrix(trues, preds))

os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Modelo guardado en {MODEL_OUT}")
