# RENOMBRA LAS IMAGENES DE UNA CARPETA

import os
import shutil

# Rutas de las carpetas
input_folder = "no_fire2_1"
output_folder = "no_fire2_1nuevo"

# Crear la carpeta de destino si no existe
os.makedirs(output_folder, exist_ok=True)

# Obtener todas las imágenes (ajusta las extensiones si es necesario)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
images = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

# Ordenar las imágenes (por nombre original)
images.sort()

# Renombrar y copiar imágenes
for i, img_name in enumerate(images, start=1):
    ext = os.path.splitext(img_name)[1]  # Obtener extensión
    new_name = f"NF{i}{ext}"  # Nuevo nombre (ej: 1.jpg, 2.png)
    shutil.copy(os.path.join(input_folder, img_name), os.path.join(output_folder, new_name))

print(f"Renombradas {len(images)} imágenes y guardadas en {output_folder}.")
