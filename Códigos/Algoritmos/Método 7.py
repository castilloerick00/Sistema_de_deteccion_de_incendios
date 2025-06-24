"""---------------------   Importacion de librerias    ------------------- """

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def rgb_to_pjf(R, G, B):                                                        # Convierte una imagen de RGB a PJF

    R = R.astype(np.float16)
    G = G.astype(np.float16)
    B = B.astype(np.float16)
    J = R - G
    F = R + G - B
    J[J < 0] = 0
    F[F < 0] = 0
    F[F > 255] = 255
    P = np.sqrt(R**2 + G**2 + B**2)
    P[P > 255] = 255

    return P, J, F

def detect_fire(R, G, B):                                                       # Detecta fuego en una imagen usando el modelo de color PJF

    P, J, F = rgb_to_pjf(R, G, B)                                               # Aplicar reglas de detección de fuego
    mask1 = P > 220
    mask2 = J > 60
    mask3 = F > 230

    fire_mask1 = mask1 & mask2
    fire_mask2 = mask2 & mask3
    fire_mask = fire_mask1 + fire_mask2

    return fire_mask

def dibujar_cajas_fuego(imagen, mascara, area_minima=10, distancia_maxima=10): # Función para dibujar cajas en los objetos detectados como fuego
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

input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/PJF'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/PJF/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

def procesar_carpeta(carpeta, contiene_fuego_esperado, f):
    for filename in sorted(os.listdir(carpeta)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            fire_mask = detect_fire(R, G, B)
            fire_mask = fire_mask.astype(np.uint8) * 255
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, fire_mask)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            f.flush()
            time.sleep(0.052)  # Esperar 30 segundos antes de procesar la siguiente imagen

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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 07.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 07.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
