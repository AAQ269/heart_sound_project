import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import FOSTER2025Loader
from src.star_pcg import STAR_PCG

# Create results folder
OUTPUT_DIR = "data/results/star_pcg_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_get(stats, key):
    """ Safe dictionary access (to avoid KeyError) """
    if stats is None:
        return None
    return stats.get(key, None)

def process_record(record):
    """ Run STAR-PCG on a single PCG recording """

    filename = record["filename"]
    signal = record["signal"]
    fs = record["fs"]
    duration = record["duration"]

    print("\n=====================================================")
    print(f"🎧 Processing {filename}  ({duration:.1f} sec)")
    print("=====================================================")

    try:
        # Initialize STAR-PCG
        star = STAR_PCG(fs=fs, verbose=False)

        init_len = min(10, duration)      # use first 10 seconds for initialization
        init_samples = int(init_len * fs)

        star.initialize(signal[:init_samples])

        # Run adaptive denoising
        clean_signal, info = star.denoise(signal)

        stats = star.get_stats()

        # 🌟 PLOT
        plot_path = os.path.join(OUTPUT_DIR, f"{filename}_starpcg.png")

        plt.figure(figsize=(14, 8))

        # Original
        plt.subplot(2, 1, 1)
        to_plot = min(3000, len(signal))
        plt.plot(signal[:to_plot], label="Original", alpha=0.7)
        plt.title(f"{filename} — Original PCG")
        plt.grid(True, alpha=0.3)

        # Cleaned
        plt.subplot(2, 1, 2)
        plt.plot(clean_signal[:to_plot], label="Cleaned (STAR-PCG)", color="green", alpha=0.7)
        plt.title("Cleaned PCG")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"📁 Saved plot: {plot_path}")

        # Return results
        return {
            "filename": filename,
            "duration_sec": duration,
            "estimated_hr_bpm": None if star.T_avg is None else 60 / star.T_avg,

            "skipped": safe_get(stats, "skipped_count"),
            "light": safe_get(stats, "light_count"),
            "heavy": safe_get(stats, "heavy_count"),

            "skipped_percent": safe_get(stats, "skipped_percent"),
            "light_percent": safe_get(stats, "light_percent"),
            "heavy_percent": safe_get(stats, "heavy_percent"),

            "avg_confidence": safe_get(stats, "avg_confidence"),
            "energy_saved_percent": safe_get(stats, "energy_saved_percent"),
        }

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return {
            "filename": filename,
            "error": str(e)
        }


def main():
    print("\n🚀 START — Running STAR-PCG on all FOSTER 2025 recordings...\n")

    loader = FOSTER2025Loader()
    dataset = loader.load_all(max_samples=None)

    print(f"\n📌 Total recordings found: {len(dataset)}\n")

    results = []

    for record in dataset:
        result = process_record(record)
        results.append(result)

    # Save results to Excel
    excel_path = os.path.join(OUTPUT_DIR, "star_pcg_summary.xlsx")
    df = pd.DataFrame(results)

    df.to_excel(excel_path, index=False)

    print("\n=====================================================")
    print(f"📊 ALL DONE! Excel summary saved at:\n➡️  {excel_path}")
    print("=====================================================")


if __name__ == "__main__":
    main()
