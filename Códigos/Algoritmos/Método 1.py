"""---------------------   Importacion de librerias    ------------------- """

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def detectar_fuego(imagen):                                                     # Función para detectar fuego en la imagen

    R, G, B = imagen[:, :, 0], imagen[:, :, 1], imagen[:, :, 2]                 # Extraer canales RGB
    R_media = np.mean(R)
    condicion1 = (R > G) & (G > B)                                              # Condiciones del modelo estadístico para detectar fuego
    condicion2 = R > R_media
    condicion3 = ((B / (R + 1)) <= 0.45)
    condicion4 = ((B / (G + 1)) <= 0.95)
    condicion5 = ((G / (R + 1)) <= 0.65) & (0.25 <= (G / (R + 1)))
    mascara_fuego = condicion1 & condicion2 & condicion3 & condicion4 & condicion5
    mascara_fuego = mascara_fuego.astype(np.uint8) * 255
    imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(imagen, mascara_fuego)    # Dibujar rectángulos alrededor del fuego

    return imagen_con_cajas, hay_fuego

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
        cv2.rectangle(imagen_con_cajas, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

""" -------------------                Código Principal               ------------------------ """

# input_folder = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color'      # Carpeta con imágenes originales
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/RGB1'      # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/RGB1/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_path = os.path.join(input_folder, filename)
#             imagen = cv2.imread(image_path)
#             imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)                    # Convertir de BGR a RGB
#             imagen_con_cajas, hay_fuego = detectar_fuego(imagen)                # Procesar imagen
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")
# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")


input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/RGB1'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/RGB1/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])


def procesar_carpeta(carpeta, contiene_fuego_esperado, f):
    for filename in sorted(os.listdir(carpeta)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            imagen = cv2.imread(image_path)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_con_cajas, hay_fuego = detectar_fuego(imagen)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            #f.flush()
            #time.sleep(0.07)  # Esperar 30 segundos antes de procesar la siguiente imagen
            
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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 01.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 01.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
