"""---------------------   Importacion de librerias    ------------------- """

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import time
import signal


"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def detect_fire_rgb(imagen_rgb):

    R, G, B = cv2.split(imagen_rgb)
    RR, RG, RB = 190, 100, 140                                                  # Definir umbrales óptimos
    condicion_1 = (R > G) & (G > B)                                             # Aplicar condiciones en RGB
    masc1 = imagen_rgb.copy()
    masc1[condicion_1 == 0] = 0
    condicion_2 = (masc1[:,:,0] > RR) & (masc1[:,:,1] > RG) & (masc1[:,:,2] < RB)
    masc2 = masc1.copy()
    masc2[condicion_2 == 0] = 0

    return condicion_2, masc2

def detect_fire_ycbcr(imagen_ycbcr):

    Y, Cr, Cb = cv2.split(imagen_ycbcr)                                         # Calcular los valores medios de Y, Cb y Cr
    Y_mean = np.mean(Y)
    Cb_mean = np.mean(Cb)
    Cr_mean = np.mean(Cr)
    condicion_3 = (Y > Cb) & (Cr > Cb)                                          # Aplicar condiciones en YCbCr
    masc3 = imagen_ycbcr.copy()
    masc3[condicion_3 == 0] = 0
    condicion_4 = (masc3[:,:,0] > Y_mean) & (masc3[:,:,1] > Cr_mean) & (masc3[:,:,2] < Cb_mean)
    masc4 = imagen_ycbcr.copy()
    masc4[condicion_4 == 0] = 0
    
    return condicion_4, masc4

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

""" -------------------                Código Principal                ------------------------ """

# input_folder = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color'      # Carpeta con imágenes originales
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/YCrCb2' # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/YCrCb2/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):

#             image_path = os.path.join(input_folder, filename)
#             img_rgb = cv2.imread(image_path)
#             imagen_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)               # Convertir de BGR a RGB
#             fire_mask_rgb, mask1 = detect_fire_rgb(imagen_rgb)                  # 1 Funcion RGB
#             imagen_ycbcr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2YCrCb)        # 2 Funcion YCbCr
#             fire_mask_ycbcr, mask2 = detect_fire_ycbcr(imagen_ycbcr)
#             fire_mask_final = fire_mask_rgb & fire_mask_ycbcr
#             mask3 = imagen_rgb.copy()
#             mask3[fire_mask_final == 0] = 0
#             coef_corr = correlation_coefficient(fire_mask_rgb, fire_mask_ycbcr) # 3 Correlacion
#             if coef_corr < 0.5:                                                 # 4. Aplicar filtro de correlación: Si la correlación es muy baja (< 0.85), descartamos detección
#                 print(f"[INFO] Alta correlación detectada ({coef_corr:.2f}). Puede ser un objeto estático.")
#                 fire_mask_final[:] = 0                                          # Borra la detección
#                 mask3[:] = 0
#             fire_mask_final = fire_mask_final.astype(np.uint8) * 255
#             imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(imagen_rgb, fire_mask_final)  # Dibujar rectángulos alrededor de las áreas de fuego detectadas
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")

# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")

input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/YCrCb2'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario1/YCrCb2/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

def procesar_carpeta(carpeta, contiene_fuego_esperado, f):                                                 # Abrir archivo de log
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            img_rgb = cv2.imread(image_path)
            imagen_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)               # Convertir de BGR a RGB
            fire_mask_rgb, mask1 = detect_fire_rgb(imagen_rgb)                  # 1 Funcion RGB
            imagen_ycbcr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2YCrCb)        # 2 Funcion YCbCr
            fire_mask_ycbcr, mask2 = detect_fire_ycbcr(imagen_ycbcr)
            fire_mask_final = fire_mask_rgb & fire_mask_ycbcr
            fire_mask_final = fire_mask_final.astype(np.uint8) * 255
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(imagen_rgb, fire_mask_final)  # Dibujar rectángulos alrededor de las áreas de fuego detectadas
            output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            f.flush()
            time.sleep(0.055)  # Esperar 30 segundos antes de procesar la siguiente imagen

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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 04.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 04.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
