import os
import numpy as np
import pandas as pd
from src.star_pcg import STAR_PCG
from scipy.io import wavfile

# المسارات
input_dir = "data/raw/PhysioNet2016/training/training-a"
output_dir = "data/star_outputs"
#label_csv = "data/raw/PhysioNet2016/training/training-a/REFERENCE.csv"
label_csv = os.path.join(input_dir, "REFERENCE.csv")
# تحميل ملف التصنيفات
label_df = pd.read_csv(label_csv, header=None, names=["id", "label"])
label_dict = dict(zip(label_df["id"], label_df["label"]))  # مثال: {'a0001': -1, 'a0002': 1}

# إنشاء مجلد المخرجات
os.makedirs(output_dir, exist_ok=True)

# معالجة كل ملف صوتي
for fname in sorted(os.listdir(input_dir)):
    if fname.endswith(".wav"):
        pid = fname.replace(".wav", "")
        fs, sig = wavfile.read(os.path.join(input_dir, fname))

        # الحصول على التصنيف
        if pid not in label_dict:
            print(f" Missing label for {pid}, skipping.")
            continue

        label = int(label_dict[pid])  # -1 or 1

        try:
            star = STAR_PCG(fs=fs, verbose=True)
            star.initialize(sig, duration=5)

            np.savez(
                os.path.join(output_dir, f"{pid}.npz"),
                cleaned=star.template_time,
                fft=star.template_freq,
                noise_std=star.noise_std,
                label=label
            )
            print(f" Saved {pid}.npz | Label = {label}")
        except Exception as e:
            print(f" Error processing {pid}: {e}")
