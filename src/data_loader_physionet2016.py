import os
import numpy as np
from scipy.io import wavfile


class PhysioNet2016Loader:
    """
    Loader for the PhysioNet / CinC Challenge 2016 PCG Dataset
    """

    def __init__(self, base_dir=r"C:\Users\USER\Desktop\Abrar\heart_sound_project\data\raw\PhysioNet2016\training"):
        self.base_dir = base_dir
        print(f" PhysioNet2016 Loader using base directory:\n{self.base_dir}")

    def load_group(self, group="training-a", max_files=None):
        """
        Loads WAV + HEA files from a specific group (training-a ... training-f)

        Returns list of dict:
        {
            'filename': ...,
            'signal': numpy array,
            'fs': sampling rate,
            'duration': seconds,
            'label': Normal/Abnormal/Unknown
        }
        """

        group_path = os.path.join(self.base_dir, group)

        if not os.path.exists(group_path):
            print(f" Group folder does not exist: {group_path}")
            return []

        print(f"\n Loading PhysioNet group: {group}")
        print(f"Folder: {group_path}")

        files = [f for f in os.listdir(group_path) if f.endswith(".wav")]

        if len(files) == 0:
            print(" No WAV files found!")
            return []

        if max_files:
            files = files[:max_files]

        data = []

        for wav_file in files:
            filename = wav_file.replace(".wav", "")
            wav_path = os.path.join(group_path, wav_file)
            hea_path = os.path.join(group_path, filename + ".hea")

            # -------- Read WAV --------
            try:
                fs, signal = wavfile.read(wav_path)
                signal = signal.astype(np.float32)

                # Normalize 16-bit PCM
                if signal.dtype != np.float32:
                    signal = signal / np.max(np.abs(signal))
            except Exception as e:
                print(f" Error reading WAV: {wav_path}")
                print(e)
                continue

            # -------- Read HEA --------
            label = "Unknown"
            duration = len(signal) / fs

            if os.path.exists(hea_path):
                try:
                    with open(hea_path, 'r') as f:
                        lines = f.readlines()

                    # labels format: "# annotation: Normal" OR "Abnormal"
                    for line in lines:
                        if "Normal" in line:
                            label = "Normal"
                        elif "Abnormal" in line:
                            label = "Abnormal"

                except:
                    pass

            data.append({
                "filename": filename,
                "signal": signal,
                "fs": fs,
                "duration": duration,
                "label": label
            })

            print(f" Loaded {filename} | fs={fs} | duration={duration:.1f}s | label={label}")

        print(f"\n Total loaded from {group}: {len(data)} recordings")
        return data
