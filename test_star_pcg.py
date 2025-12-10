from src.star_pcg import STAR_PCG
from src.data_loader import FOSTER2025Loader

loader = FOSTER2025Loader()
data = loader.load_all(max_samples=1)

signal = data[0]['signal']
fs = data[0]['fs']

star = STAR_PCG(fs=fs)
star.initialize(signal[:2000])
clean, info = star.denoise(signal)

print("Test finished.")
