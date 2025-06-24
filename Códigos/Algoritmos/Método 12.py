

"""---------------------   Importacion de librerias    ------------------- """

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

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

def detect_fire(image, R, G, B):                                                       # Detecta fuego en una imagen usando el modelo de color PJF

    P, J, F = rgb_to_pjf(R, G, B)                                               # Aplicar reglas de detección de fuego
    mask1 = P > 220
    mask2 = J > 60
    mask3 = F > 230

    fire_mask1 = mask1 & mask2
    fire_mask2 = mask2 & mask3
    fire_mask = fire_mask1 + fire_mask2

    ################################ YCrCb-2 ################################
    R, G, B = cv2.split(image)
    RR, RG, RB = 190, 100, 140                                                  # Definir umbrales óptimos
    condicion_1 = (R > G) & (G > B)                                             # Aplicar condiciones en RGB
    masc1 = image.copy()
    masc1[condicion_1 == 0] = 0
    condicion_2 = (masc1[:,:,0] > RR) & (masc1[:,:,1] > RG) & (masc1[:,:,2] < RB)
    masc2 = masc1.copy()
    masc2[condicion_2 == 0] = 0

    imagen_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
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

    fire_mask_fy = condicion_2 * condicion_4
    fire_mask_fy = fire_mask_fy.astype(np.uint8) * 255

    fire_mask_f2 = fire_mask_fy & fire_mask
    fire_mask_f2 = fire_mask_f2.astype(np.uint8) * 255
    ################################ YCrCb-2 ################################


    return fire_mask_f2



"""---------------------       Funciones para enmarcar las zonas de fuego       ------------------- """


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

input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Escenario3_1/condicion1/F'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Escenario3_1/condicion1/NF'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Escenario3/Color/condicion1'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Escenario3/Color/condicion1/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

def procesar_carpeta(carpeta, contiene_fuego_esperado, f):
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]  # OpenCV usa BGR
            f2 = detect_fire(image, R, G, B)
            f2 = f2.astype(np.uint8) * 255
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, f2)# Dibujar rectángulos alrededor del fuego
            output_path = os.path.join(output_folder, filename)           # Guardar imagen procesada
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            f.flush()
            time.sleep(0.028)  # Esperar 30 segundos antes de procesar la siguiente imagen

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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 12.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 12.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
