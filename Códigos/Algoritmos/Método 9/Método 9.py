# CON LECTURA DE METRICAS ED RENDIMIENTO


import os
import sys
import subprocess
from ultralytics import YOLO
from PIL import Image

# Forzar UTF-8 en la salida
sys.stdout.reconfigure(encoding='utf-8')

# Rutas
model_path = "best25.pt"
input_folders = {
    'fire': '/home/pi/TESIS/Codigos/Imagenes/Escenario3_1/condicion3/F',
    'no_fire': '/home/pi/TESIS/Codigos/Imagenes/Escenario3_1/condicion3/NF'
}
output_folder = "/home/pi/TESIS/Codigos/Imagenes/Resultados/Escenario3/IA"
os.makedirs(output_folder, exist_ok=True)

# Cargar modelo
model = YOLO(model_path)

# Iniciar captura de métricas
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

# Estructura de resultados
resultados = {
    'fire': {'correctas': [], 'incorrectas': []},
    'no_fire': {'correctas': [], 'incorrectas': []}
}

# Procesar carpetas
for folder_type, input_folder in input_folders.items():
    if not os.path.exists(input_folder):
        print(f"Advertencia: Carpeta {input_folder} no encontrada")
        continue

    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(input_folder, img_name)
        try:
            with Image.open(img_path) as img:
                img.verify()

            results = model.predict(source=img_path, conf=0.18)

            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                output_path = os.path.join(output_folder, img_name)
                im.save(output_path)

            fuego_detectado = any(len(res.boxes) > 0 for res in results)

            if folder_type == 'fire':
                cat = 'correctas' if fuego_detectado else 'incorrectas'
            else:
                cat = 'correctas' if not fuego_detectado else 'incorrectas'

            resultados[folder_type][cat].append(img_name)

        except Exception as e:
            print(f"ERROR procesando {img_name}: {e}")
            continue

# Guardar resultados en TXT
txt_path = os.path.join(output_folder, "deteccion_fuego.txt")
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("Resultados de Detección de Fuego:\n\n")
    f.write("Imágenes con FUEGO:\n")
    for img in resultados['fire']['correctas']:
        f.write(f"{img}: Sí (Detección Correcta)\n")
    for img in resultados['fire']['incorrectas']:
        f.write(f"{img}: No (Detección Errónea)\n")
    f.write("\nImágenes SIN FUEGO:\n")
    for img in resultados['no_fire']['correctas']:
        f.write(f"{img}: No (Detección Correcta)\n")
    for img in resultados['no_fire']['incorrectas']:
        f.write(f"{img}: Sí (Detección Errónea)\n")

# Finalizar captura de métricas y renombrar archivo
metricas_proc.terminate()
metricas_proc.wait()

# Renombrar archivo de métricas
archivo_original_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.csv'
archivo_nuevo_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 09.csv'
archivo_original_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
archivo_nuevo_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 09.png'

if os.path.exists(archivo_original_csv):
    os.rename(archivo_original_csv, archivo_nuevo_csv)
if os.path.exists(archivo_original_png):
    os.rename(archivo_original_png, archivo_nuevo_png)

print("Proceso completado correctamente. Resultados y métricas guardadas.")

