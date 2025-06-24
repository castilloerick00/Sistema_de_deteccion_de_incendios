"""---------------------   Importacion de librerias    ------------------- """

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def detectar_fuego(ruta_imagen, mostrar_resultado=True):


    H_MIN, H_MAX, H_MIN2, H_MAX2 = 0, 0.1, 0.9, 1                                                        # Definir umbrales de color para el fuego en HSV
    S_MIN, S_MAX = 0.65, 1.0
    V_MIN, V_MAX = 0.5, 1.0

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen en {ruta_imagen}")

    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    H_MIN, H_MAX, H_MIN2, H_MAX2 = int(H_MIN * 180), int(H_MAX * 180), int(H_MIN2 * 180), int(H_MAX2 * 180)                           # Normalizar valores de HSV
    S_MIN, S_MAX = int(S_MIN * 255), int(S_MAX * 255)
    V_MIN, V_MAX = int(V_MIN * 255), int(V_MAX * 255)

    mascara1 = cv2.inRange(imagen_hsv, (H_MIN, S_MIN, V_MIN), (H_MAX, S_MAX, V_MAX))  # Normalizar valores de HSV para OpenCV
    mascara2 = cv2.inRange(imagen_hsv, (H_MIN2, S_MIN, V_MIN), (H_MAX2, S_MAX, V_MAX))  # Normalizar valores de HSV para OpenCV
    mascara_fuego = cv2.bitwise_or(mascara1, mascara2)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara_fuego)

    r, g, b = cv2.split(resultado)
    mask = b > 175
#    mask_final = mask & mascara_fuego
    mask_final = mascara_fuego
    mask_final = mask_final.astype(np.uint8) * 255
    resultado[mask_final == 0] = 0

    return imagen_rgb, mask_final, resultado


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

# input_folder = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/'      # Carpeta con imágenes originales
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/HSV/'      # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/HSV/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_path = os.path.join(input_folder, filename)
#             imagen_rgb, mask_final, resultado = detectar_fuego(image_path)
#             imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(imagen_rgb, mask_final)# Dibujar rectángulos alrededor del fuego
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")

# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")

input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/HSV'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/HSV/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])


def procesar_carpeta(carpeta, contiene_fuego_esperado, f):                                               # Abrir archivo de log
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            imagen_rgb, mask_final, resultado = detectar_fuego(image_path)
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(imagen_rgb, mask_final)# Dibujar rectángulos alrededor del fuego
            output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
            f.write(f"{filename}: {estado_fuego}\n")
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            f.flush()
            time.sleep(0.07)  # Esperar 30 segundos antes de procesar la siguiente imagen

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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 06.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 06.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
