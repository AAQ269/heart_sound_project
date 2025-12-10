from src.data_loader_physionet2016 import PhysioNet2016Loader

loader = PhysioNet2016Loader()

data = loader.load_group("training-a", max_files=3)

print("\n===== SUMMARY =====")
for d in data:
    print(d['filename'], d['fs'], d['duration'], d['label'])