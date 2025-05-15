import os
import cv2
from pathlib import Path

def crop_textures(input_dir, output_dir, size=(128, 128), step=128):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path) or category == "patches":
            continue  # pomiń folder patches

        out_category_path = os.path.join(output_dir, category)
        Path(out_category_path).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Nie udało się wczytać: {img_path}")
                continue

            h, w = img.shape
            idx = 0
            for y in range(0, h - size[1] + 1, step):
                for x in range(0, w - size[0] + 1, step):
                    patch = img[y:y + size[1], x:x + size[0]]
                    patch_filename = f"{filename.split('.')[0]}_patch_{idx}.png"
                    cv2.imwrite(os.path.join(out_category_path, patch_filename), patch)
                    idx += 1

# Uruchom wycinanie
crop_textures("C:\\Users\Adam\\Desktop\\STUDIA_POI\\POI\\LABORATORIUM\\LAB_3", "patches", size=(128, 128), step=128)
