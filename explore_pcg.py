import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/USER/Desktop/Abrar/heart_sound_project/data/raw/FOSTER_dataset_CSV/sub001.csv")

time = df["Time"].values
pcg = df["PCG"].values

# نرسم الإشارة على شكل مقاطع كل 5 ثواني
fs = int(1 / (time[1] - time[0]))  # 10000 Hz
window = 5 * fs

for i in range(0, len(pcg), window):
    segment = pcg[i:i+window]

    plt.figure(figsize=(12, 3))
    plt.plot(segment)
    plt.title(f"Segment {i//window}  —  (time {i/fs:.1f}s to {(i+window)/fs:.1f}s)")
    plt.grid(True)
    plt.show()

    # نوقف إذا تبينك تختارين window بنفسك
    inp = input("Continue? (y/n): ")
    if inp.lower() != "y":
        break
