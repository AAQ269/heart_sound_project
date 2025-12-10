import os
import numpy as np
import matplotlib.pyplot as plt

from star_pcg import STAR_PCG
from data_loader import FOSTER2025Loader
from data_loader_physionet2016 import PhysioNet2016Loader


# ----------------------------------------------------------
# Function: Plot Template
# ----------------------------------------------------------
def plot_template(template, fs, save_path):
    plt.figure(figsize=(10,4))
    plt.plot(template, color="purple", linewidth=2)
    plt.title("Template Heart Cycle", fontsize=14)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------------------------------------
# Generate Templates for FOSTER Dataset
# ----------------------------------------------------------
def generate_templates_foster():
    print("\nGenerating templates for FOSTER dataset...")

    loader = FOSTER2025Loader("data/raw/FOSTER_dataset_CSV")
    data = loader.load_all()

    print(f"Loaded {len(data)} recordings\n")

    save_dir = "data/results/templates_foster"
    os.makedirs(save_dir, exist_ok=True)

    for item in data:
        signal = item["signal"]
        fs = item["fs"]
        name = item["filename"]

        print(f" Processing {name}...")

        # NEW: shorten initialization to 3 seconds max
        init_samples = min(len(signal), 3 * fs)
        init_signal = signal[:init_samples]

        pcg = STAR_PCG(fs=fs, verbose=False)
        pcg.initialize(init_signal)

        # get template
        template = pcg.template_time

        save_path = os.path.join(save_dir, f"{name}_template.png")
        plot_template(template, fs, save_path)

        print(f" Saved template to: {save_path}")



# ----------------------------------------------------------
# Generate Templates for PhysioNet Dataset
# ----------------------------------------------------------
def generate_templates_physionet():
    print("\n=== PhysioNet 2016 ===\n")

    # المسار الأساسي
    loader = PhysioNet2016Loader("data/raw/PhysioNet2016/training")

    print(f"PhysioNet2016 Loader using base directory:")
    print(loader.base_dir)

    # تحميل البيانات باستخدام الدالة الصحيحة
    data = loader.load_group(max_files=None) 

    print(f"Total loaded: {len(data)} recordings\n")

    save_dir = "data/results/templates_physionet"
    os.makedirs(save_dir, exist_ok=True)

    # معالجة كل مريض
    for item in data:
        fname = item['filename']
        signal = item['signal']
        fs = item['fs']

        print(f"Processing {fname}...")

        # إنشاء STAR-PCG
        pcg = STAR_PCG(fs=fs, verbose=False)

        # فقط مرحلة initialization للحصول على template
        pcg.initialize(signal)

        template = pcg.template_time
        if template is None:
            print(f" Failed to extract template for {fname}")
            continue

        save_path = os.path.join(save_dir, f"{fname}_template.png")
        plot_template(template, fs, save_path)

        print(f"Saved template: {save_path}")



# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    print("Generating templates for both datasets...\n")
    #generate_templates_foster()
    generate_templates_physionet()
    print("ALL DONE!")
