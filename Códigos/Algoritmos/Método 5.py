"""---------------------   Importacion de librerias    ------------------- """

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def hsi_to_rgb(hsi_image):                                                      # Convierte una imagen HSI a RGB con corrección

    H, S, I = cv2.split(hsi_image)
    H = H * 2 * np.pi                                                           # Convertir a radianes
    I = I * 255.0                                                               # Desnormalizar intensidad
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    mask1 = (H >= 0) & (H < 2 * np.pi / 3)                                      # Caso 1: 0° ≤ H < 120°
    B[mask1] = I[mask1] * (1 - S[mask1])
    R[mask1] = I[mask1] * (1 + S[mask1] * np.cos(H[mask1]) / np.cos(np.pi / 3 - H[mask1]))
    G[mask1] = 3 * I[mask1] - (R[mask1] + B[mask1])

    mask2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)                          # Caso 2: 120° ≤ H < 240°
    H2 = H[mask2] - 2 * np.pi / 3
    R[mask2] = I[mask2] * (1 - S[mask2])
    G[mask2] = I[mask2] * (1 + S[mask2] * np.cos(H2) / np.cos(np.pi / 3 - H2))
    B[mask2] = 3 * I[mask2] - (R[mask2] + G[mask2])

    mask3 = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)                              # Caso 3: 240° ≤ H < 360°
    H3 = H[mask3] - 4 * np.pi / 3
    G[mask3] = I[mask3] * (1 - S[mask3])
    B[mask3] = I[mask3] * (1 + S[mask3] * np.cos(H3) / np.cos(np.pi / 3 - H3))
    R[mask3] = 3 * I[mask3] - (G[mask3] + B[mask3])

    R = np.clip(R, 0, 255)                                                      # Limitar valores al rango [0, 255]
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    image_rgb = cv2.merge([R, G, B]).astype(np.uint8)                           # Combinar canales en una imagen

    return image_rgb

def rgb_to_hsi(image):                                                          # Convierte una imagen RGB a HSI

    img = image.astype(np.float32)
    R, G, B = cv2.split(img)
    R1 = R/(R+G+B)
    G1 = G/(R+G+B)
    B1 = B/(R+G+B)
    I = (R + G + B) / (3)                                                       # Cálculo de la Intensidad (I)
    I1 = I/255

    min_RGB1 = np.minimum(np.minimum(R1, G1), B1)                               # Cálculo de la Saturación (S)
    S1 = np.abs(1 - 3 * min_RGB1)                                               # Añadir epsilon para evitar división por cero
    S1[np.isnan(S1)] = 0
    S = S1 * 100

    num = 0.5 * ((R1 - G1) + (R1 - B1))                                         # Cálculo del Tono (H)
    den = np.sqrt((R1 - G1)**2 + (R1 - B1) * (G1 - B1))                         # Añadir epsilon para evitar división por cero
    theta = np.arccos(num / den)
    H1 = np.where(B1 > G1, 2 * np.pi - theta, theta)                            # Ajustar ángulo en función de B y G
    H1 = H1 / (2 * np.pi)                                                       # Normalizar de 0 a 1
    H1[np.isnan(H1)] = 0
    H = H1 * 2 * np.pi                                                          # radianes
    H = H * 180 / np.pi                                                         # grados
    image_hsi =  np.stack([H1, S1, I1], axis=-1)                                # Forma (h, w, 3)

    return H1, S1, I1, image_hsi

def apply_rgb_hsi_criteria(image):                                              # Aplica el criterio RGB-HSI para detección de llamas

    H, S, I, image_hsi = rgb_to_hsi(image)                                      # Convertir imagen a escala de grises para binarización
    R, G, B = cv2.split(image)                                                  # Umbrales basados en las ecuaciones del paper
    R_t = 125                                                                   # Umbral de R (ajustable según la iluminación)
    S_t = 0.2                                                                   # Umbral de saturación
    K = 0.15                                                                    # Umbras 0.15 - 2

    mask = (R > G) & (G > B) & (R > R_t) & (S > 0.1) & (S > (255 - R) / 20) & (S >= (255 - R) * S_t / R_t)  # Aplicar criterio original RGB-HSI
    mask1 = (R > B) & (G > B) & (R >= R_t) & (S >= (255 - R) * np.mean(S) * K)
    mask2 = mask & mask1
    flame_mask = np.zeros_like(R)                                               # Convertir la máscara binaria a imagen
    flame_mask[mask2] = 255

    # # ---- VISUALIZACIÓN ----
    # fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # axes[0].imshow(H)
    # axes[0].set_title("H")
    # axes[1].imshow(S)
    # # axes[2].imshow(cv2.cvtColor(mask2, cv2.COLOR_YCrCb2RGB))
    # axes[1].set_title("S")
    # axes[2].imshow(I)
    # axes[2].set_title("I")

    return flame_mask, H, S, I, image_hsi, R, G, B

def apply_morphological_expansion(binary_image, kernel_size=3):                 # Aplica expansión morfológica para unir píxeles de llama dispersos

    if kernel_size not in [3, 5]:
        raise ValueError("El tamaño del kernel debe ser 3 o 5 según el paper")
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)) # Crear un elemento estructurante circular
    expanded_image = cv2.dilate(binary_image, struct_element, iterations=1)     # Aplicar operación de expansión

    return expanded_image

def dibujar_cajas_fuego(imagen, mascara, area_minima=100, distancia_maxima=10): # Función para dibujar cajas en los objetos detectados como fuego
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagen_con_cajas = imagen.copy()
    rectangulos = []

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w * h >= area_minima:                                                # Filtrar objetos demasiado pequeños
            rectangulos.append((x, y, w, h))

    rectangulos_fusionados = fusionar_rectangulos(rectangulos, distancia_maxima)
    hay_fuego = len(rectangulos_fusionados) > 0                                 # Si no hay rectángulos válidos, no hay fuego

    for x, y, w, h in rectangulos_fusionados:
        cv2.rectangle(imagen_con_cajas, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return imagen_con_cajas, hay_fuego

def fusionar_rectangulos(rectangulos, distancia_maxima):                        # Función para fusionar rectángulos cercanos
    if not rectangulos:
        return []
    rectangulos = sorted(rectangulos, key=lambda r: (r[0], r[1]))
    fusionados = []
    x0, y0, w0, h0 = rectangulos[0]

    for x, y, w, h in rectangulos[1:]:
        if (x - (x0 + w0) <= distancia_maxima) and (y - (y0 + h0) <= distancia_maxima):
            x0 = min(x0, x)
            y0 = min(y0, y)
            w0 = max(x + w - x0, w0)
            h0 = max(y + h - y0, h0)
        else:
            fusionados.append((x0, y0, w0, h0))
            x0, y0, w0, h0 = x, y, w, h

    fusionados.append((x0, y0, w0, h0))
    return fusionados

""" -------------------                Código Principal                ------------------------ """

# input_folder = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color'      # Carpeta con imágenes originales
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/HSI' # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/HSI/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)               # Convertir de BGR a RGB
#             flame_mask, H, S, I, image_hsi, R, G, B = apply_rgb_hsi_criteria(image)         # Aplicar criterio de detección de llamas
#             image_rgb = hsi_to_rgb(image_hsi)
#             expanded_mask = apply_morphological_expansion(flame_mask, kernel_size=3)        # Aplicar expansión morfológica con ventana de 3x3
#             salida = image.copy()
#             salida[expanded_mask == 0] = 0
#             imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, expanded_mask.astype(np.uint8) * 255)  # Dibujar rectángulos alrededor de las áreas de fuego detectadas
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")

# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")


input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/HSI'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/HSI/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])


def procesar_carpeta(carpeta, contiene_fuego_esperado, f):                                               # Abrir archivo de log
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)               # Convertir de BGR a RGB
            flame_mask, H, S, I, image_hsi, R, G, B = apply_rgb_hsi_criteria(image)         # Aplicar criterio de detección de llamas
            image_rgb = hsi_to_rgb(image_hsi)
            expanded_mask = apply_morphological_expansion(flame_mask, kernel_size=3)        # Aplicar expansión morfológica con ventana de 3x3
            salida = image.copy()
            salida[expanded_mask == 0] = 0
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, expanded_mask.astype(np.uint8) * 255)  # Dibujar rectángulos alrededor de las áreas de fuego detectadas
            output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            f.flush()
            time.sleep(0.05)  # Esperar 30 segundos antes de procesar la siguiente imagen

try:
    with open(log_file, "w") as f:
        f.write("Resultados de Detección de Fuego:\n\n")
        f.write("Imágenes con FUEGO:\n")
        procesar_carpeta(input_folder_fire, True, f)
        f.write("\nImágenes SIN fuego:\n")
        procesar_carpeta(input_folder_no_fire, False, f)
finally:
    # Terminar el proceso de metricas_lineal.py
    metricas_proc.terminate()
    metricas_proc.wait()

#Ruta del archivo actual y nuevo nombre
archivo_original = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.csv'
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 05.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 05.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
