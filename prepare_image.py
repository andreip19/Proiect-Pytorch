import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

INPUT_DIR = "./test_images/"
OUTPUT_DIR = "./bin_images/"

def process_directory():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists():
        print(f"Eroare: Directorul {INPUT_DIR} nu exista. Adauga fisiere .png in el.")
        return

    for img_path in input_path.glob("*.png"):
        try:
           
            image = Image.open(img_path).convert("L")

     
            pixels = np.array(image)
            corners = [int(pixels[0, 0]), int(pixels[0, -1]), int(pixels[-1, 0]), int(pixels[-1, -1])]
            if sum(corners) / len(corners) > 127:
                image = ImageOps.invert(image)

          
            image = image.point(lambda p: p if p > 20 else 0)
            bbox = image.getbbox()

            if bbox is None:
                print(f"Sarit {img_path.name}: Nu a fost detectata nicio cifra in imagine.")
                continue

            image = image.crop(bbox)

           
            image.thumbnail((20, 20), Image.Resampling.LANCZOS)

           
            img_array = np.array(image)
            mass = np.sum(img_array)

            if mass == 0:
                cy, cx = image.height // 2, image.width // 2
            else:
                y_indices, x_indices = np.indices(img_array.shape)
                cy = np.sum(y_indices * img_array) / mass
                cx = np.sum(x_indices * img_array) / mass

        
            canvas = Image.new("L", (28, 28), 0)
            x_offset = int(np.round(14.0 - cx))
            y_offset = int(np.round(14.0 - cy))

            canvas.paste(image, (x_offset, y_offset))

          
            array = np.array(canvas, dtype=np.uint8)
            out_filename = img_path.stem + ".bin"
            out_filepath = Path(OUTPUT_DIR) / out_filename
            array.tofile(out_filepath)

            print(f"Imagine pregatita (Center of Mass): {out_filepath}")
            
        except Exception as e:
            print(f"Eroare la procesarea {img_path.name}: {e}")

if __name__ == "__main__":
    process_directory()