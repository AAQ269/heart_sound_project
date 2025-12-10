import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════
# الإعدادات
# ═══════════════════════════════════════════════════

DATA_PATH = "data/outputs/tcn_data/"
BATCH_SIZE = 16
EPOCHS = 150
PATIENCE = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════
# Focal Loss للتعامل مع Class Imbalance
# ═══════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss - يركز على الأمثلة الصعبة
    alpha: وزن للكلاس النادر
    gamma: معامل التركيز (كلما زاد، زاد التركيز على الصعب)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ═══════════════════════════════════════════════════
# Data Augmentation Functions
# ═══════════════════════════════════════════════════

def augment_signal(signal, noise_level=0.02):
    """إضافة ضوضاء"""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def time_shift(signal, shift_max=50):
    """إزاحة زمنية"""
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=-1)

def scale_amplitude(signal, scale_range=(0.8, 1.2)):
    """تغيير السعة"""
    scale = np.random.uniform(*scale_range)
    return signal * scale

def add_dropout_noise(signal, dropout_rate=0.05):
    """إضافة dropout عشوائي"""
    mask = np.random.rand(*signal.shape) > dropout_rate
    return signal * mask

def augment_dataset_balanced(X, y, target_samples_per_class=300):
    """
    Augmentation متوازن - يعادل توزيع الكلاسات
    """
    class_counts = Counter(y)
    print(f"\n Original class distribution:")
    for class_label, count in class_counts.items():
        print(f"   Class {class_label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    X_balanced = []
    y_balanced = []
    
    for class_label in np.unique(y):
        # استخرج عينات هذا الكلاس
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]
        
        current_count = len(X_class)
        needed_samples = target_samples_per_class - current_count
        augment_factor = max(1, needed_samples // current_count + 1)
        
        print(f"\n Class {class_label} augmentation:")
        print(f"   Current: {current_count}")
        print(f"   Target: {target_samples_per_class}")
        print(f"   Augmentation factor: {augment_factor}x")
        
        # أضف العينات الأصلية
        for signal in X_class:
            X_balanced.append(signal)
            y_balanced.append(class_label)
        
        # Augment لحد ما نوصل للهدف
        augmented = 0
        while len([y for y in y_balanced if y == class_label]) < target_samples_per_class:
            for signal in X_class:
                if len([y for y in y_balanced if y == class_label]) >= target_samples_per_class:
                    break
                
                aug_signal = signal.copy()
                
                # طبق transformations عشوائية متعددة
                transformations_applied = []
                
                if np.random.rand() > 0.3:
                    aug_signal = augment_signal(aug_signal, noise_level=np.random.uniform(0.01, 0.03))
                    transformations_applied.append("noise")
                
                if np.random.rand() > 0.3:
                    aug_signal = time_shift(aug_signal, shift_max=np.random.randint(30, 80))
                    transformations_applied.append("shift")
                
                if np.random.rand() > 0.4:
                    aug_signal = scale_amplitude(aug_signal, scale_range=(0.75, 1.25))
                    transformations_applied.append("scale")
                
                if np.random.rand() > 0.6:
                    aug_signal = add_dropout_noise(aug_signal, dropout_rate=0.03)
                    transformations_applied.append("dropout")
                
                X_balanced.append(aug_signal)
                y_balanced.append(class_label)
                augmented += 1
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    final_counts = Counter(y_balanced)
    print(f"\n Balanced class distribution:")
    for class_label, count in final_counts.items():
        print(f"   Class {class_label}: {count} samples ({count/len(y_balanced)*100:.1f}%)")
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    return X_balanced[indices], y_balanced[indices]

# ═══════════════════════════════════════════════════
# 1. تحميل البيانات
# ═══════════════════════════════════════════════════

print("\n Loading data...")
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

print(f"Original X shape: {X.shape}")
print(f"Original y shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

# تحويل labels
y[y == -1] = 0

print(f"Data shape: {X.shape}")

# ═══════════════════════════════════════════════════
# 2. تقسيم البيانات
# ═══════════════════════════════════════════════════

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print(f"\nDataset split:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ═══════════════════════════════════════════════════
# 3. Balanced Augmentation
# ═══════════════════════════════════════════════════

print("\n" + "="*60)
print(" Applying Balanced Data Augmentation")
print("="*60)

X_train_aug, y_train_aug = augment_dataset_balanced(
    X_train, y_train, 
    target_samples_per_class=300  # كل كلاس يصير 300 sample
)

print(f"\n Final training set size: {len(X_train_aug)}")

# ═══════════════════════════════════════════════════
# 4. تحويل لـ Tensors
# ═══════════════════════════════════════════════════

X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val_tensor, y_val_tensor), 
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor), 
    batch_size=BATCH_SIZE
)

# ═══════════════════════════════════════════════════
# 5. بناء النموذج
# ═══════════════════════════════════════════════════

from tcn_model.model import HeavyTCN

input_channels = X_train_tensor.shape[1]
sequence_length = X_train_tensor.shape[2]

print("\n" + "="*60)
print(" Building Model")
print("="*60)
print(f"  Input channels: {input_channels}")
print(f"  Sequence length: {sequence_length}")

model = HeavyTCN(input_channels=input_channels, num_classes=2).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# ═══════════════════════════════════════════════════
# 6. Loss Function & Optimizer
# ═══════════════════════════════════════════════════

criterion = FocalLoss(alpha=0.75, gamma=2.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8
)

print(f"\n Training Configuration:")
print(f"  Loss: Focal Loss (alpha=0.75, gamma=2.0)")
print(f"  Optimizer: AdamW (lr=5e-5, weight_decay=1e-4)")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {EPOCHS}")
print(f"  Early stopping patience: {PATIENCE}")

# ═══════════════════════════════════════════════════
# 7. Training Loop
# ═══════════════════════════════════════════════════

print("\n" + "="*60)
print(" Training Started")
print("="*60)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    # ─────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += yb.size(0)
        train_correct += predicted.eq(yb).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # ─────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += yb.size(0)
            val_correct += predicted.eq(yb).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # ─────────────────────────────────────────────────
    # Early Stopping
    # ─────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f" Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break

# ═══════════════════════════════════════════════════
# 8. التقييم النهائي
# ═══════════════════════════════════════════════════

print("\n" + "="*60)
print("Final Evaluation on Test Set")
print("="*60)

model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        outputs = model(xb)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

test_acc = accuracy_score(all_labels, all_preds)
print(f"\n Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# حساب metrics لكل كلاس
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

print(f"\n Per-Class Metrics:")
for i in range(len(precision)):
    print(f"  Class {i}:")
    print(f"    Precision: {precision[i]:.4f}")
    print(f"    Recall:    {recall[i]:.4f}")
    print(f"    F1-Score:  {f1[i]:.4f}")
    print(f"    Support:   {support[i]}")

# ═══════════════════════════════════════════════════
# 9. Visualizations
# ═══════════════════════════════════════════════════

os.makedirs("data/results", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss curves
axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy curves
axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2)
axes[0, 1].plot(val_accs, label='Val Acc', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            cbar_kws={'label': 'Count'})
axes[1, 0].set_xlabel('Predicted', fontsize=12)
axes[1, 0].set_ylabel('True', fontsize=12)
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 4. Confidence distribution
all_probs = np.array(all_probs)
max_probs = all_probs.max(axis=1)
axes[1, 1].hist(max_probs, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(max_probs.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {max_probs.mean():.3f}')
axes[1, 1].set_xlabel('Confidence Score', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data/results/training_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Plots saved to: data/results/training_results.png")

# ═══════════════════════════════════════════════════
# 10. حفظ النتائج
# ═══════════════════════════════════════════════════

results = {
    'test_accuracy': float(test_acc),
    'train_accuracy': float(train_accs[-1]),
    'val_accuracy': float(val_accs[-1]),
    'best_val_loss': float(best_val_loss),
    'epochs_trained': len(train_losses),
    'class_0_precision': float(precision[0]),
    'class_0_recall': float(recall[0]),
    'class_0_f1': float(f1[0]),
    'class_1_precision': float(precision[1]),
    'class_1_recall': float(recall[1]),
    'class_1_f1': float(f1[1]),
    'confusion_matrix': cm.tolist()
}

import json
with open('data/results/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(" Results saved to: data/results/training_results.json")

print("\n" + "="*60)
print(" Training Complete!")
print("="*60)
print(f"\n Final Results Summary:")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Class 0 F1-Score: {f1[0]:.4f}")
print(f"  Class 1 F1-Score: {f1[1]:.4f}")
print(f"  Model saved: models/best_model.pth")