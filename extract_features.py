import cv2
import os
import numpy as np

DATASET_FOLDER = "dataset"

def extract_features(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))  # Resize to standard shape
    return image.flatten() / 255.0         # Normalize and flatten

features = []
image_paths = []

for filename in os.listdir(DATASET_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(DATASET_FOLDER, filename)
        print(f"Processing: {path}")
        features.append(extract_features(path))
        image_paths.append(path)

features = np.array(features)
np.save("features.npy", features)
np.save("paths.npy", image_paths)

print("✅ Feature extraction complete. Saved to features.npy and paths.npy")
