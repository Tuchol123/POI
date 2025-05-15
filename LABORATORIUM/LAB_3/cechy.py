import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
import pandas as pd

DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
FEATURES = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

def extract_features_from_patch(img, distances, angles):
    img = (img / 4).astype(np.uint8)  # Redukcja do 64 poziomów szarości
    glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)

    feature_vector = []
    for prop in FEATURES:
        vals = graycoprops(glcm, prop)
        feature_vector.extend(vals.flatten())
    return feature_vector

def extract_all_features(patch_dir):
    data = []
    for category in os.listdir(patch_dir):
        category_path = os.path.join(patch_dir, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = imread(img_path, as_gray=True)
            img = (img * 255).astype(np.uint8)

            features = extract_features_from_patch(img, DISTANCES, ANGLES)
            features.append(category)  # label
            data.append(features)
    return data

# Wyciągnij cechy z patchy i zapisz jako CSV
features = extract_all_features("patches")
columns = [f"{feat}_{d}_{int(np.rad2deg(a))}" for feat in FEATURES for d in DISTANCES for a in ANGLES]
columns.append("label")

df = pd.DataFrame(features, columns=columns)
df.to_csv("features.csv", index=False)
print("Zapisano features.csv")
