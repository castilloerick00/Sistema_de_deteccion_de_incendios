"""---------------------   Importacion de librerias    ------------------- """

import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import os
import subprocess
import time
import signal


"""---------------------       Funciones para el procesamiento de la imagen       ------------------- """

def preprocess_image(img_ycbcr):
    Y, Cr, Cb = cv2.split(img_ycbcr)                                            # Aplicar reglas heurísticas
    heuristic_mask = (Y > Cb) & (Cr > Cb)
    imagen1 = img_ycbcr.copy()
    imagen1[heuristic_mask == 0] = 0
    Y, Cr, Cb = cv2.split(imagen1)

    return Y, Cb, Cr, imagen1

def fuzzy_fire_classification(Y, Cb, Cr):                                       # Sistema de lógica difusa para clasificar píxeles de fuego

    diff_Y_Cb = ctrl.Antecedent(np.linspace(-1, 1, 100), 'diff_Y_Cb')           # 1️ Definir las variables difusas
    diff_Cr_Cb = ctrl.Antecedent(np.linspace(-1, 1, 100), 'diff_Cr_Cb')
    Pf = ctrl.Consequent(np.linspace(0, 1, 100), 'Pf')

    diff_Y_Cb['NB'] = fuzz.trapmf(diff_Y_Cb.universe, [-1, -1, -0.3, 0])        # 2️ Definir funciones de membresía más detalladas
    diff_Y_Cb['NS'] = fuzz.trimf(diff_Y_Cb.universe, [-0.3, -0.2, -0.05])
    diff_Y_Cb['ZE'] = fuzz.trimf(diff_Y_Cb.universe, [-0.05, 0, 0.05])
    diff_Y_Cb['PS'] = fuzz.trimf(diff_Y_Cb.universe, [0.05, 0.2, 0.3])
    diff_Y_Cb['PB'] = fuzz.trapmf(diff_Y_Cb.universe, [0, 0.3, 1, 1])
    diff_Cr_Cb['NB'] = fuzz.trapmf(diff_Cr_Cb.universe, [-1, -1, -0.1, 0.1])
    diff_Cr_Cb['NS'] = fuzz.trimf(diff_Cr_Cb.universe, [0, 0.1, 0.2])
    diff_Cr_Cb['ZE'] = fuzz.trimf(diff_Cr_Cb.universe, [0.1, 0.2, 0.3])
    diff_Cr_Cb['PS'] = fuzz.trimf(diff_Cr_Cb.universe, [0.2, 0.3, 0.4])
    diff_Cr_Cb['PB'] = fuzz.trapmf(diff_Cr_Cb.universe, [0.3, 0.7, 1, 1])
    Pf['NB'] = fuzz.trimf(Pf.universe, [-0.16, 0.01, 0.18])
    Pf['NS'] = fuzz.trimf(Pf.universe, [-0.12, 0.05, 0.22])
    Pf['ZE'] = fuzz.trimf(Pf.universe, [0, 0.24, 0.5])
    Pf['PS'] = fuzz.trimf(Pf.universe, [0.13, 0.36, 0.58])
    Pf['PB'] = fuzz.trimf(Pf.universe, [0.7, 1, 1])

    rules = [                                                                   # 3️ Definir las reglas difusas completas según la tabla del paper
        ctrl.Rule(diff_Y_Cb['NB'] & diff_Cr_Cb['NB'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NB'] & diff_Cr_Cb['NS'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NB'] & diff_Cr_Cb['ZE'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NB'] & diff_Cr_Cb['PS'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NB'] & diff_Cr_Cb['PB'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NS'] & diff_Cr_Cb['NB'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NS'] & diff_Cr_Cb['NS'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NS'] & diff_Cr_Cb['ZE'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NS'] & diff_Cr_Cb['PS'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['NS'] & diff_Cr_Cb['PB'], Pf['NS']),
        ctrl.Rule(diff_Y_Cb['ZE'] & diff_Cr_Cb['NB'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['ZE'] & diff_Cr_Cb['NS'], Pf['NB']),
        ctrl.Rule(diff_Y_Cb['ZE'] & diff_Cr_Cb['ZE'], Pf['NS']),
        ctrl.Rule(diff_Y_Cb['ZE'] & diff_Cr_Cb['PS'], Pf['NS']),
        ctrl.Rule(diff_Y_Cb['ZE'] & diff_Cr_Cb['PB'], Pf['ZE']),
        ctrl.Rule(diff_Y_Cb['PS'] & diff_Cr_Cb['NB'], Pf['NS']),
        ctrl.Rule(diff_Y_Cb['PS'] & diff_Cr_Cb['NS'], Pf['ZE']),
        ctrl.Rule(diff_Y_Cb['PS'] & diff_Cr_Cb['ZE'], Pf['ZE']),
        ctrl.Rule(diff_Y_Cb['PS'] & diff_Cr_Cb['PS'], Pf['PS']),
        ctrl.Rule(diff_Y_Cb['PS'] & diff_Cr_Cb['PB'], Pf['PB']),
        ctrl.Rule(diff_Y_Cb['PB'] & diff_Cr_Cb['NB'], Pf['NS']),
        ctrl.Rule(diff_Y_Cb['PB'] & diff_Cr_Cb['NS'], Pf['ZE']),
        ctrl.Rule(diff_Y_Cb['PB'] & diff_Cr_Cb['ZE'], Pf['PS']),
        ctrl.Rule(diff_Y_Cb['PB'] & diff_Cr_Cb['PS'], Pf['PB']),
        ctrl.Rule(diff_Y_Cb['PB'] & diff_Cr_Cb['PB'], Pf['PB']),
    ]

    Pf_ctrl = ctrl.ControlSystem(rules)                                         # 4️ Crear el sistema de control difuso
    Pf_sim = ctrl.ControlSystemSimulation(Pf_ctrl)
    fire_prob = np.zeros_like(Y, dtype=np.float32)                              # 5️ Evaluar cada píxel de la imagen
    diff_Y_Cb_values = (Y.astype(np.float32) - Cb.astype(np.float32)) / 255.0
    diff_Cr_Cb_values = (Cr.astype(np.float32) - Cb.astype(np.float32)) / 255.0
    coords = np.array(np.meshgrid(range(Y.shape[0]), range(Y.shape[1]))).T.reshape(-1, 2)   # Convertimos la imagen a una lista de tuplas

    for x, y in coords:                                                         # Evaluar de forma optimizada
        Pf_sim.input['diff_Y_Cb'] = diff_Y_Cb_values[x, y]
        Pf_sim.input['diff_Cr_Cb'] = diff_Cr_Cb_values[x, y]
        Pf_sim.compute()
        fire_prob[x, y] = Pf_sim.output['Pf']

    return fire_prob

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

# input_folder = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color1/'      # Carpeta con imágenes originales
# output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/YCrCb1/'      # Carpeta para guardar imágenes procesadas
# os.makedirs(output_folder, exist_ok=True)                                       # Crear la carpeta de salida si no existe
# log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/YCrCb1/deteccion_fuego.txt' # Archivo de salida con detecciones

# with open(log_file, "w") as f:                                                  # Abrir archivo de log
#     for filename in sorted(os.listdir(input_folder)):                           # Procesar imágenes en orden
#         if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#             image_path = os.path.join(input_folder, filename)
#             img_rgb = cv2.imread(image_path)
#             img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
#             Y, Cb, Cr, imagen1 = preprocess_image(img_ycbcr)                                 # Paso 1: Preprocesamiento
#             fire_prob_map = fuzzy_fire_classification(Y, Cb, Cr)                    # Paso 2: Aplicación de lógica difusa
#             binary_fire_mask = (fire_prob_map >= 0.5)  # 1 = Fuego, 0 = No fuego
#             binary_fire_mask = binary_fire_mask.astype(np.uint8) * 255
#             final = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
#             # final[binary_fire_mask == 0] = 0
#             imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(final, binary_fire_mask)# Dibujar rectángulos alrededor del fuego
#             output_path = os.path.join(output_folder, filename)                 # Guardar imagen procesada
#             cv2.imwrite(output_path, cv2.cvtColor(imagen_con_cajas, cv2.COLOR_RGB2BGR))
#             estado_fuego = "Sí" if hay_fuego else "No"                          # Guardar resultado en el archivo de texto
#             f.write(f"{filename}: {estado_fuego}\n")

# print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")

input_folder_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color1/fire2'
input_folder_no_fire = '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color1/no_fire2'
output_folder = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario2/YCrCb1'
os.makedirs(output_folder, exist_ok=True)
log_file = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Color/Escenario2/YCrCb1/deteccion_fuego.txt'

# Lanzar metricas_lineal.py en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])


def procesar_carpeta(carpeta, contiene_fuego_esperado, f):                                               # Abrir archivo de log
    for filename in sorted(os.listdir(carpeta)):                           # Procesar imágenes en orden
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(carpeta, filename)
            img_rgb = cv2.imread(image_path)
            img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
            Y, Cb, Cr, imagen1 = preprocess_image(img_ycbcr)                                 # Paso 1: Preprocesamiento
            fire_prob_map = fuzzy_fire_classification(Y, Cb, Cr)                    # Paso 2: Aplicación de lógica difusa
            binary_fire_mask = (fire_prob_map >= 0.5)  # 1 = Fuego, 0 = No fuego
            binary_fire_mask = binary_fire_mask.astype(np.uint8) * 255
            final = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            # final[binary_fire_mask == 0] = 0
            imagen_con_cajas, hay_fuego = dibujar_cajas_fuego(final, binary_fire_mask)# Dibujar rectángulos alrededor del fuego
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
archivo_nuevo = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 03.csv'
archivo_original1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo1 = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 03.png'

# Renombrar el archivo
os.rename(archivo_original, archivo_nuevo)
os.rename(archivo_original1, archivo_nuevo1)

print("Procesamiento completado. Imágenes guardadas y detecciones registradas.")
