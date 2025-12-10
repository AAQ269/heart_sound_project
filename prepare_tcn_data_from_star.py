# prepare_tcn_data_from_star.py

import os
import numpy as np
import json
import shutil

INPUT_DIR = "data/star_outputs"
OUTPUT_DIR = "data/outputs/tcn_data"
META_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_file(file_path, output_dir, metadata):
    base_name = os.path.basename(file_path).replace(".npz", "")
    npz = np.load(file_path)

    # Load the signals and label
    signal = npz["cleaned"]
    label = npz["label"]  # -1: normal, 1: abnormal

    # Save as .npy
    signal_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(signal_path, signal)

    metadata[base_name] = {
        "label": int(label),
        "length": len(signal)
    }

def main():
    print("Preparing TCN data from STAR-PCG outputs...")
    ensure_dir(OUTPUT_DIR)

    metadata = {}

    for file in os.listdir(INPUT_DIR):
        if file.endswith(".npz"):
            file_path = os.path.join(INPUT_DIR, file)
            process_file(file_path, OUTPUT_DIR, metadata)

    # Save metadata
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f" Done. Saved {len(metadata)} files to: {OUTPUT_DIR}")
    print(f" Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
