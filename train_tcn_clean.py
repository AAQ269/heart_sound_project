import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from model_tcn_attention import SETCN

# =========================================================
# 0) الإعدادات العامة
# =========================================================

DATA_PATH = "data/outputs/tcn_data/"
BATCH_SIZE = 20
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 3e-4
PATIENCE = 12
TARGET_LEN = 1024

print(f"Using device: {DEVICE}")


# =========================================================
# 1) إصلاح تحميل البيانات (بدون قص مدمر)
# =========================================================

def load_and_fix_signal(path, target_len=1024):
    sig = np.load(path)

    if len(sig) >= target_len:
        return sig[:target_len]
    else:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(sig)] = sig
        return padded


def load_dataset():
    meta_path = os.path.join(DATA_PATH, "metadata.json")
    import json
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    X, y = [], []
    for fname, meta in metadata.items():
        sig_path = os.path.join(DATA_PATH, f"{fname}.npy")
        if not os.path.exists(sig_path):
            continue

        sig = load_and_fix_signal(sig_path, TARGET_LEN)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)  # normalization
        X.append(sig)
        y.append(0 if meta["label"] == -1 else 1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # reshape: (N, 1, L)
    X = X[:, np.newaxis, :]

    return X, y


print("\nLoading dataset...")
X, y = load_dataset()
print(f"Dataset loaded: X={X.shape}, y={y.shape}")


# =========================================================
# 2) Augmentation صحي (بدون تخريب الـ PCG)
# =========================================================

def augment_signal_clean(x):
    # noise خفيف فقط
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, 0.01, x.shape)

    # scale بسيط
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale

    return x


def apply_augmentation(X, y):
    X_aug, y_aug = [], []
    for sig, lbl in zip(X, y):
        X_aug.append(sig)
        y_aug.append(lbl)

        # نسخة augment
        aug = augment_signal_clean(sig.copy())
        X_aug.append(aug)
        y_aug.append(lbl)

    return np.array(X_aug), np.array(y_aug)


print("\nApplying clean augmentation...")
X, y = apply_augmentation(X, y)
print(f"After augmentation: {X.shape}")


# =========================================================
# 3) تقسيم البيانات
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)


# تحويل إلى Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

print("\nDataset split done.")


# =========================================================
# 4) نموذج MediumTCN (الأفضل هنا)
# =========================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=pad, dilation=dilation)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out[:, :, :res.size(2)] + res)


class MediumTCN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        layers = []
        channels = [32, 32, 32]
        for i in range(len(channels)):
            dilation = 2 ** i
            in_c = in_channels if i == 0 else channels[i - 1]
            out_c = channels[i]
            layers.append(TemporalBlock(in_c, out_c, 3, dilation))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]
        return self.fc(x)


#model = MediumTCN().to(DEVICE)
#print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
model = SETCN().to(DEVICE)


# =========================================================
# 5) Loss + Optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


# =========================================================
# 6) Training Loop (محسن + Early Stopping)
# =========================================================

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()

    train_acc = correct / len(X_train)

    # validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == yb).sum().item()

    val_acc = val_correct / len(X_val)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_tcn.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered!")
            break


# =========================================================
# 7) Final Test Evaluation
# =========================================================

model.load_state_dict(torch.load("best_tcn.pth"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")
