#save_asl_classes.py
import numpy as np
import os

DATASET_DIR = "asl_dataset/train"  

class_names = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

np.save(f"asl_model_classes.npy", class_names)
print("Classes saved:", class_names)