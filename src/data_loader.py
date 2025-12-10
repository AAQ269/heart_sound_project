# src/data_loader.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly


class FOSTER2025Loader:
    """
    Loader V4:
    - Read PCG only
    - Take first 5 seconds
    - Resample using resample_poly (best for biosignals)
    - Normalize
    """

    def __init__(self, data_dir="C:\\Users\\USER\\Desktop\\Abrar\\heart_sound_project\\data\\raw\\FOSTER_dataset_CSV"):
        self.data_dir = Path(data_dir)
        self.target_fs = 1000          # final FS
        self.window_seconds = 5        # first 5 seconds only
        print(f"Dataset directory: {self.data_dir}")

    def load_all(self, max_samples=None):
        print(" Loading FOSTER 2025 Dataset...")

        files = list(self.data_dir.glob("*.csv"))
        print(f"Found {len(files)} CSV files")

        all_data = []

        for i, file in enumerate(files):
            if max_samples and i >= max_samples:
                break

            df = pd.read_csv(file)

            if "PCG" not in df.columns or "Time" not in df.columns:
                print(f"Skipping {file.name} (missing PCG or Time)")
                continue

            # READ PCG + TIME
            time = df["Time"].values
            pcg = df["PCG"].values

            # ORIGINAL FS
            dt = time[1] - time[0]
            original_fs = int(1 / dt)  # = 10000 Hz

            # TAKE FIRST 5 SECONDS
            raw_count = original_fs * self.window_seconds
            pcg_segment = pcg[:raw_count]

            # RESAMPLE USING resample_poly
            down_factor = original_fs // self.target_fs     # 10000/1000 = 10
            pcg_resampled = resample_poly(pcg_segment, up=1, down=down_factor)

            # NORMALIZE
            if np.max(np.abs(pcg_resampled)) > 0:
                pcg_resampled = pcg_resampled / np.max(np.abs(pcg_resampled))

            entry = {
                "signal": pcg_resampled,
                "filename": file.name,
                "fs": self.target_fs,
                "duration": self.window_seconds
            }

            all_data.append(entry)

        print("\n📊 Dataset Statistics:")
        print(f"   Total recordings: {len(all_data)}")
        print(f"   Duration: {self.window_seconds} seconds")
        print(f"   Resampled to: {self.target_fs} Hz")

        return all_data
