from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import glob
import subprocess
import time

# Configuración
CATEGORIAS = {
    'positivo': {
        'folder': '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire',
        'display_name': 'con FUEGO',
        'deteccion_esperada': 'Sí'
    },
    'negativo': {
        'folder': '/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire',
        'display_name': 'SIN FUEGO',
        'deteccion_esperada': 'No'
    }
}

# Cargar modelo
model = load_model('modelo_deteccion_incendios28.h5')

# Predicción
def predecir_incendio(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 192))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img)[0][0]

# Ruta archivo resultados
archivo_resultados = '/home/pi/TESIS/Codigos/Imagenes/Resultados/IA/Escenario1/CNN/deteccion_fuego.txt'

# Ejecutar script de métricas en segundo plano
metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

try:
    with open(archivo_resultados, 'w', encoding='utf-8') as f:
        f.write("Resultados de Detección de Fuego:\n\n")

        for tipo, config in CATEGORIAS.items():
            correctos = 0
            total = 0
            resultados_seccion = []

            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for path in glob.glob(os.path.join(config['folder'], ext)):
                    prediccion = predecir_incendio(path)
                    if prediccion is None:
                        continue

                    nombre_archivo = os.path.basename(path)
                    deteccion = "Sí" if prediccion > 0.15 else "No"
                    es_correcto = deteccion == config['deteccion_esperada']
                    estado = "Correcta" if es_correcto else "Errónea"
                    emoji = "✅" if es_correcto else "❌"

                    resultados_seccion.append(f"{nombre_archivo}: {deteccion} (Detección {estado})")
                    if es_correcto:
                        correctos += 1
                    total += 1

            f.write(f"Imágenes {config['display_name']}:\n")
            f.write('\n'.join(resultados_seccion) + '\n\n')

finally:
    # Terminar el proceso de métricas
    metricas_proc.terminate()
    metricas_proc.wait()

    # Renombrar archivo CSV y gráfico
    archivo_original_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.csv'
    archivo_nuevo_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 08.csv'
    archivo_original_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
    archivo_nuevo_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 08.png'

    if os.path.exists(archivo_original_csv):
        os.rename(archivo_original_csv, archivo_nuevo_csv)
    if os.path.exists(archivo_original_png):
        os.rename(archivo_original_png, archivo_nuevo_png)

print("Análisis completado. Resultados y métricas guardados.")

