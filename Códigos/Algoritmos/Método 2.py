"""---------------------   Importacion de librerias    ------------------- """

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import time
import signal

"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def extract_roi(image, image_path):
    R, G, B = cv2.split(image)                                                  # Separar en canales
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    mean_R = np.exp(np.mean(np.log(R + 1)))                                     # Calculamos media geométrica de cada canal
    mean_G = np.exp(np.mean(np.log(G + 1)))
    mean_B = np.exp(np.mean(np.log(B + 1)))

    R_new = np.maximum(R - mean_R, 0)                                           # Resta de c/canal con su media
    G_new = np.maximum(G - mean_G, 0)
    B_new = np.maximum(B - mean_B, 0)

    image_new = cv2.imread(image_path)                                          # Copiamos la imagen original
    image_new[:,:,0] = R_new
    image_new[:,:,1] = G_new
    image_new[:,:,2] = B_new

    mask = np.ones_like(R, dtype=np.uint8)                                      # Primer filtrado
    mask[(R_new < B_new) | (R_new + B_new < G_new)] = 0

    filtered = image_new.copy()                                                 # Filtrado de cada canal
    filtered = filtered.astype(np.uint16)
    filtered[:, :, 0] *= mask
    filtered[:, :, 1] *= mask
    filtered[:, :, 2] *= mask

    adjusted_R = 2 * (filtered[:, :, 0] - filtered[:, :, 2])                    # Filtrado adicional en el canal rojo
    adjusted_R = np.clip(adjusted_R, 0, 255).astype(np.uint8)

    threshold_value, binary_mask = cv2.threshold(adjusted_R, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Umbralización de Otsu

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)    # Remover pequeños objetos (ruido)
    min_size = 8                                                                # Mínimo de píxeles para considerar una región válida
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):                                              # Omitir fondo
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255

    return cleaned_mask

def algoritmo3(image):

    tau1 = 195  # Valor predeterminado para tau1
    tau2 = 5    # Valor predeterminado para tau2

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]                          # Calculamos los histogramas de c/canal
    hist_R = cv2.calcHist([R], [0], None, [256], [0, 256]).flatten()
    hist_G = cv2.calcHist([G], [0], None, [256], [0, 256]).flatten()
    hist_B = cv2.calcHist([B], [0], None, [256], [0, 256]).flatten()

    candidate_tau1 = np.where((hist_R > hist_G) & (hist_G > hist_B))[0]         # Paso 1: Identificar valores candidatos para τ1 (Rojo > Verde > Azul)
    if len(candidate_tau1) > 0:
        tmp = candidate_tau1[0]
        for i in range(1, len(candidate_tau1)):
            if candidate_tau1[i] < tmp + 10:
                tmp = candidate_tau1[i]
            else:
                tau1 = candidate_tau1[i]
                break
            if i == len(candidate_tau1)-1:
                if candidate_tau1[-1] > 100:
                    tau1 = np.sum(candidate_tau1) / len(candidate_tau1)         # Promedio
                else:
                    tau1 = candidate_tau1[-1]

    candidate_tau2 = np.where((hist_B > hist_G) & (hist_B > hist_R))[0]         # Paso 2: Identificar valores candidatos para τ2 (Azul > Verde > Rojo)
    if len(candidate_tau2) > 0:
        tmp = candidate_tau2[0]
        for i in range(1, len(candidate_tau2)):                                 # Itera desde i = 2 hasta sizeOf(η)
            if candidate_tau2[i] < tmp + 5:
                tmp = candidate_tau2[i - 1]                                     # tmp ← η(i-1)
            else:
                tau2 = candidate_tau2[i]                                        # τ2 ← η(i)
                break
            if i == len(candidate_tau2)-1:
                tau2 = candidate_tau2[-1]                                       # Se toma el último valor

    if tau2 < 3:                                                                # Corrección de τ2 según lo propuesto en el paper
        tau2 = 40
    elif tau2 < 5:
        tau2 *= 10
    elif tau2 < 10:
        tau2 *= 5
    else:
        tau2 = 5

    return int(tau1), int(tau2)

def dividir_franjas(image):

    tau1_as, tau2_as = algoritmo3(image)                                        # Paso 1: División en bandas horizontales y verticales (3 y 3)
    height, width, _ = image.shape
    horizontal_stripes = [image[int(i * height / 3):int((i + 1) * height / 3), :] for i in range(3)]
    vertical_stripes = [image[:, int(i * width / 3):int((i + 1) * width / 3)] for i in range(3)]
    stripes = horizontal_stripes + vertical_stripes

    return stripes, tau1_as, tau2_as

def algoritmo2(stripes, tau1_as, tau2_as):

    processed_stripes = []
    for stripe in stripes:
        tau1, tau2 = algoritmo3(stripe)                                         # Calculamos T1 y T2 para cada franja
        gray = cv2.cvtColor(stripe, cv2.COLOR_RGB2GRAY)
        mu = np.mean(gray)
        epsilon = (255 + (np.sqrt(np.sum((gray - mu) ** 2) / (gray.size - 1)))) / 255
        delta = tau1_as / (tau2_as * 2) if epsilon < tau1_as / (tau2_as * 2) else epsilon
        tau1 = tau1_as if tau1_as > tau1 else tau1                              # Ajustar τ1 según la condición
        stripe = stripe.astype(np.int16)
        # Generar imagen binaria usando g(x, y)
        binary_image = np.where((tau1 > stripe[:, :, 0]) & (stripe[:, :, 2] >= stripe[:, :, 0]), 0,
                        np.where((stripe[:, :, 2] > stripe[:, :, 1]) & (stripe[:, :, 2] >= stripe[:, :, 0]), 0,
                        # np.where((stripe[:, :, 1] + (stripe[:, :, 2]) >= 0.8*stripe[:, :, 0]), 0,
                        np.where((stripe[:, :, 1] + stripe[:, :, 2] > stripe[:, :, 0]) & (delta * stripe[:, :, 2] > stripe[:, :, 1]), 0, 1)))
        binary_image = binary_image.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        for i in range(1, num_labels):                                          # Remover objetos pequeños
            if stats[i, cv2.CC_STAT_AREA] < 8:
                binary_image[labels == i] = 0
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)) # Rellenar regiones pequeñas
        masked_stripe = stripe.copy()                                           # Aplicar máscara a la imagen original
        masked_stripe[binary_image == 0] = 0
        processed_stripes.append(masked_stripe)

    merged_horizontal = np.concatenate(processed_stripes[:3], axis=0)           # Concatenamos franjas horizontales
    merged_vertical = np.concatenate(processed_stripes[3:], axis=1)             # Concatenamos franjas verticales

    return merged_horizontal, merged_vertical

def algoritmo4(merged_horizontal, merged_vertical):

    r_h, c_h, _ = merged_horizontal.shape
    f = np.zeros((r_h, c_h))
    merged_horizontal = merged_horizontal.astype(np.float32)                    # Primer filtrado de c/imagen
    merged_vertical = merged_vertical.astype(np.float32)
    img1 = np.where((merged_horizontal[:, :, 1] + merged_horizontal[:, :, 2] > merged_horizontal[:, :, 0]), 0,
            np.where((100 > merged_horizontal[:, :, 0]), 0 , 255))
    img2 = np.where((merged_vertical[:, :, 1] + merged_vertical[:, :, 2] > merged_vertical[:, :, 0]), 0,
            np.where((100 > merged_vertical[:, :, 0]), 0 , 255))
    p_h = (img1 > 0).astype(int)                                                # Segundo filtrado de c/imagen
    p_v = (img2 > 0).astype(int)
    f = p_h * p_v                                                               # Interseccion entre las dos imagenes

    return f

def dibujar_cajas_fuego(imagen, mascara, area_minima=50, distancia_maxima=10): # Función para dibujar cajas en los objetos detectados como fuego
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
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/RGB2/'      # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/RGB2/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_path = os.path.join(input_folder, filename)
#             image = cv2.imread(image_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             mask1 = extract_roi(image, image_path)
#             stripes, tau1_as, tau2_as = dividir_franjas(image)
#             merged_horizontal, merged_vertical = algoritmo2(stripes, tau1_as, tau2_as)
#             mask2 = algoritmo4(merged_horizontal, merged_vertical)
#             mask_final = mask1 & mask2
#             mask_final = mask_final.astype(np.uint8) * 255
#             imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, mask_final)# Dibujar rectángulos alrededor del fuego
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")

# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")


input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire2'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire2'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario2/RGB2'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario2/RGB2/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

def procesar_carpeta(carpeta, contiene_fuego_esperado, f):
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask1 = extract_roi(image, image_path)
            stripes, tau1_as, tau2_as = dividir_franjas(image)
            merged_horizontal, merged_vertical = algoritmo2(stripes, tau1_as, tau2_as)
            mask2 = algoritmo4(merged_horizontal, merged_vertical)
            mask_final = mask1 & mask2
            mask_final = mask_final.astype(np.uint8) * 255
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(image, mask_final)# Dibujar rectángulos alrededor del fuego
            output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
            cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
            estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
            resultado = "Correcta" if hay_fuego == contiene_fuego_esperado else "Errónea"
            f.write(f"{filename}: {estado_fuego} (Detección {resultado})\n")
            #f.flush()
            #time.sleep(1)  # Esperar 30 segundos antes de procesar la siguiente imagen

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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 02.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 02.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")


