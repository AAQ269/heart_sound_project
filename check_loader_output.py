from src.data_loader import FOSTER2025Loader
import matplotlib.pyplot as plt

loader = FOSTER2025Loader()
data = loader.load_all(max_samples=1)

signal = data[0]["signal"]
fs = data[0]["fs"]

plt.figure(figsize=(12,4))
plt.plot(signal)
plt.title("Signal AFTER Loader (resampled + window)")
plt.grid(True)
plt.show()
