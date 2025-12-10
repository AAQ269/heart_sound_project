# run_star_pcg_physionet.py

import os
import matplotlib.pyplot as plt
from src.data_loader_physionet2016 import PhysioNet2016Loader
from src.star_pcg import STAR_PCG
import numpy as np

SAVE_DIR = "data/results/star_pcg_physionet"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("\n Running STAR-PCG on PhysioNet 2016...\n")

    loader = PhysioNet2016Loader(
        base_dir="data/raw/PhysioNet2016/training"
    )

    #  training-a
    #data = loader.load_wavs(max_files=3)
    data = loader.load_group("training-a", max_files=3)

    if len(data) == 0:
        print(" No data found!")
        return

    for sample in data:
        filename = sample['filename']
        signal = sample['signal']
        fs = sample['fs']

        print(f"\n===============================================")
        print(f" Processing: {filename}")
        print(f"===============================================\n")

        # STEP 1 — Initialize STAR-PCG
        star = STAR_PCG(fs=fs, verbose=True)

        init_duration = min(10, sample['duration'])
        init_samples = int(init_duration * fs)
        star.initialize(signal[:init_samples])

        # STEP 2 — Denoise full signal
        clean_signal, info = star.denoise(signal)

        # STEP 3 — Save plot
        plt.figure(figsize=(16, 8))

        N = int(5 * fs)  # أول 5 ثواني

        plt.subplot(2, 1, 1)
        plt.plot(signal[:N], alpha=0.7)
        plt.title(f"{filename} — Original PCG")
        plt.grid(alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(clean_signal[:N], color='green', alpha=0.7)
        plt.title("Cleaned PCG")
        plt.grid(alpha=0.3)

        out_path = os.path.join(SAVE_DIR, f"{filename}_starpcg.png")
        plt.savefig(out_path, dpi=120)
        plt.close()

        print(f" Saved plot: {out_path}")

        # Print stats
        stats = star.get_stats()
        if stats:
            print("\n Stats:")
            print(stats)

    print("\n ALL DONE! Results saved in:", SAVE_DIR)

if __name__ == "__main__":
    main()
