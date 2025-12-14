import os
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from model_tcn_attention import SETCN   # Import model

DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

DATA_PATH = "data/outputs/tcn_data/"
TARGET_LEN = 1024
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 10
LR = 3e-4


# ============================================================
# 1) Load dataset
# ============================================================

def load_and_fix_signal(path, target_len=1024):
    sig = np.load(path)

    if len(sig) >= target_len:
        return sig[:target_len]
    else:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(sig)] = sig
        return padded


def load_dataset():
    import json
    meta_path = os.path.join(DATA_PATH, "metadata.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    X, y = [], []

    for fname, info in meta.items():
        path = os.path.join(DATA_PATH, f"{fname}.npy")
        if not os.path.exists(path):
            continue

        sig = load_and_fix_signal(path, TARGET_LEN)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

        X.append(sig)
        y.append(0 if info["label"] == -1 else 1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    X = X[:, np.newaxis, :]
    return X, y


print("\nLoading dataset...")
X, y = load_dataset()
print("Dataset:", X.shape, y.shape)

# ============================================================
# 2) Train / Val / Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

print("Dataset split done.")

# ============================================================
# 3) Create FP32 model for QAT
# ============================================================

fp32_model = SETCN()
fp32_model.train()

print("\nBase FP32 model parameters:", sum(p.numel() for p in fp32_model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(fp32_model.parameters(), lr=LR)

# ============================================================
# 4) Prepare model for QAT
# ============================================================

fp32_model.qconfig = tq.get_default_qat_qconfig('fbgemm')
tq.prepare_qat(fp32_model, inplace=True)

print("\nQAT preparation done.")

# ============================================================
# 5) Training loop with early stopping
# ============================================================

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):

    fp32_model.train()
    train_loss = 0
    correct = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = fp32_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()

    train_acc = correct / len(X_train)

    # Validation
    fp32_model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            out = fp32_model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()

    val_acc = val_correct / len(X_val)

    print(f"Epoch {epoch+1}/{EPOCHS} | FP32 Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(fp32_model.state_dict(), "best_tcn_qat_fp32.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break


# ============================================================
# 6) Convert saved FP32-QAT → INT8
# ============================================================

print("\nConverting model to INT8...")

# إعادة بناء نموذج QAT
qat_model = SETCN()
qat_model.qconfig = tq.get_default_qat_qconfig('fbgemm')
tq.prepare_qat(qat_model, inplace=True)

# تحميل الأوزان الخاصة بـ QAT
qat_model.load_state_dict(torch.load("best_tcn_qat_fp32.pth"))

# التحويل إلى INT8
int8_model = tq.convert(qat_model.eval(), inplace=False)
torch.save(int8_model.state_dict(), "tcn_int8.pth")

print("INT8 model saved as tcn_int8.pth")

# ============================================================
# 7) Evaluate INT8 model
# ============================================================

print("\nTesting INT8 model...")

int8_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        out = int8_model(xb)
        preds = out.argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

print("\nINT8 Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

acc = accuracy_score(all_labels, all_preds)
print(f"INT8 Test Accuracy: {acc*100:.2f}%")
