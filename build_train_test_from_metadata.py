import os
import numpy as np
import json
from sklearn.model_selection import train_test_split

# المسارات
DATA_DIR = "data/outputs/tcn_data/"
META_FILE = os.path.join(DATA_DIR, "metadata.json")

# إعدادات
SEQUENCE_LENGTH = 1024  # عدد النقاط لكل عينة، اضبطيه حسب البيانات
TEST_SIZE = 0.2         # نسبة البيانات للاختبار

def load_metadata(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

def load_and_process_sample(file_path, target_length=1024):
    signal = np.load(file_path)

    # قص أو تعبئة الإشارة لتكون بطول ثابت
    if len(signal) > target_length:
        signal = signal[:target_length]
    elif len(signal) < target_length:
        padding = target_length - len(signal)
        signal = np.pad(signal, (0, padding), mode='constant')
    
    return signal

def main():
    print("🔧 Preparing training and test datasets...")
    
    metadata = load_metadata(META_FILE)
    
    X = []
    y = []

    for filename, info in metadata.items():
        signal_path = os.path.join(DATA_DIR, f"{filename}.npy")
        if not os.path.exists(signal_path):
            continue

        signal = load_and_process_sample(signal_path, SEQUENCE_LENGTH)
        X.append(signal)
        y.append(1 if info["label"] == 1 else 0)  # تحويل التصنيف إلى 0/1
    
    X = np.array(X)
    y = np.array(y)

    print(f" Loaded {len(X)} samples. Shape: {X.shape}")

    # إعادة تشكيل للإشارات لتناسب TCN: (N, T, C)
    X = X.reshape((X.shape[0], SEQUENCE_LENGTH, 1))

    # تقسيم Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

    # حفظ الملفات
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    print(" Saved X_train.npy, y_train.npy, X_test.npy, y_test.npy")

if __name__ == "__main__":
    main()
