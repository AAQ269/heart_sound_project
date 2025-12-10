# generate_X_y_from_metadata.py

import os
import numpy as np
import json

INPUT_DIR = "data/outputs/tcn_data"
META_PATH = os.path.join(INPUT_DIR, "metadata.json")
X_PATH = os.path.join(INPUT_DIR, "X.npy")
Y_PATH = os.path.join(INPUT_DIR, "y.npy")

TARGET_LENGTH = 1024  # الطول المطلوب لكل إشارة

def pad_or_crop(signal, target_len):
    if len(signal) < target_len:
        # نعمل padding بإضافة أصفار في النهاية
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(signal)] = signal
        return padded
    else:
        # نقص الإشارة من المنتصف
        start = (len(signal) - target_len) // 2
        return signal[start:start+target_len]

def main():
    print("Loading metadata...")
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    X_list = []
    y_list = []

    print("Building X and y arrays...")
    for fname, meta in metadata.items():
        signal_path = os.path.join(INPUT_DIR, f"{fname}.npy")
        if not os.path.exists(signal_path):
            continue

        signal = np.load(signal_path)
        signal_fixed = pad_or_crop(signal, TARGET_LENGTH)

        X_list.append(signal_fixed)
        y_list.append(meta["label"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # reshape to [samples, channels=1, length]
    X = X[:, np.newaxis, :]

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    print("Done. Saved X.npy and y.npy")

if __name__ == "__main__":
    main()
