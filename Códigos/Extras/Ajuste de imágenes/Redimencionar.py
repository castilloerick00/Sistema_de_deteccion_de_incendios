# REDIMENSIONA LAS IMAGENE DE UNA CARPETA

import os
from PIL import Image

# Ruta de la carpeta de entrada y salida
input_folder = 'Img_pruebas2_color/fire2'
output_folder = 'Img_pruebas2_color/fire3'
new_size = (250, 250)  # Tamaño deseado (ancho, alto)

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Recorrer todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                img_resized.save(output_path)
                print(f"Redimensionada y guardada: {output_path}")
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

