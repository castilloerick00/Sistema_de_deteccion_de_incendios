import cv2
import os
import subprocess

def detectar_y_guardar(ruta_imagen, ruta_cascade, carpeta_resultados, es_positivo):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"[ERROR] No se pudo cargar: {ruta_imagen}")
        return 0, None

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(ruta_cascade)
    if cascade.empty():
        print("[ERROR] Clasificador en cascada vacío")
        return 0, None

    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in objects:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    nombre_archivo = os.path.basename(ruta_imagen)
    ruta_guardado = os.path.join(carpeta_resultados, nombre_archivo)
    cv2.imwrite(ruta_guardado, imagen)

    if es_positivo and len(objects) > 0:
        return 1, f"{nombre_archivo}: Sí (Detección Correcta)"
    elif not es_positivo and len(objects) == 0:
        return 1, f"{nombre_archivo}: No (Detección Correcta)"
    else:
        if es_positivo:
            return 0, f"{nombre_archivo}: No (Detección Errónea)"
        else:
            return 0, f"{nombre_archivo}: Sí (Detección Errónea)"

def procesar_carpeta(carpeta, ruta_cascade, carpeta_resultados, es_positivo):
    total = 0
    aciertos = 0
    resultados = []
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_imagen = os.path.join(carpeta, archivo)
            resultado, linea = detectar_y_guardar(ruta_imagen, ruta_cascade, carpeta_resultados, es_positivo)
            if linea:
                resultados.append(linea)
            aciertos += resultado
            total += 1
    return aciertos, total, resultados

if __name__ == "__main__":
    ruta_cascade = "cascada/cascade.xml"
    carpeta_resultados = "/home/pi/TESIS/Codigos/Imagenes/Resultados/IA/Escenario1/HAAR1"
    ruta_log = os.path.join(carpeta_resultados, "deteccion_fuego.txt")
    os.makedirs(carpeta_resultados, exist_ok=True)

    # Iniciar captura de métricas
    metricas_proc = subprocess.Popen(['python3', '/home/pi/TESIS/Codigos/Metricas/metricas_lineal1.py'])

    try:
        aciertos_pos, total_pos, resultados_pos = procesar_carpeta(
            "/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/fire",
            ruta_cascade, carpeta_resultados, True
        )
        aciertos_neg, total_neg, resultados_neg = procesar_carpeta(
            "/home/pi/TESIS/Codigos/Imagenes/Img_pruebas_color/no_fire",
            ruta_cascade, carpeta_resultados, False
        )

        total_imagenes = total_pos + total_neg
        total_aciertos = aciertos_pos + aciertos_neg

        porcentaje_pos = (aciertos_pos / total_pos * 100) if total_pos > 0 else 0
        porcentaje_neg = (aciertos_neg / total_neg * 100) if total_neg > 0 else 0
        porcentaje_total = (total_aciertos / total_imagenes * 100) if total_imagenes > 0 else 0

        with open(ruta_log, "w", encoding="utf-8") as log:
            log.write("Resultados de Detección de Fuego:\n\n")
            log.write("Imágenes con FUEGO:\n")
            for linea in resultados_pos:
                log.write(linea + "\n")

            log.write("\nImágenes SIN FUEGO:\n")
            for linea in resultados_neg:
                log.write(linea + "\n")

            log.write("\nResumen:\n")
            log.write(f"Imágenes con fuego: {total_pos} | Aciertos: {aciertos_pos} | Éxito: {porcentaje_pos:.2f}%\n")
            log.write(f"Imágenes sin fuego: {total_neg} | Aciertos: {aciertos_neg} | Éxito: {porcentaje_neg:.2f}%\n")
            log.write(f"Total imágenes procesadas: {total_imagenes} | Aciertos totales: {total_aciertos} | Éxito global: {porcentaje_total:.2f}%\n")

        # Mostrar por consola
        print(f"Porcentaje de éxito positivo: {porcentaje_pos:.2f}%")
        print(f"Porcentaje de éxito negativo: {porcentaje_neg:.2f}%")
        print(f"Porcentaje de éxito global: {porcentaje_total:.2f}%")

    finally:
        # Terminar proceso de métricas
        metricas_proc.terminate()
        metricas_proc.wait()

        # Renombrar archivos de métricas
        archivo_original_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.csv'
        archivo_nuevo_csv = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 10.csv'
        archivo_original_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png'
        archivo_nuevo_png = '/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/Metodo 10.png'

        if os.path.exists(archivo_original_csv):
            os.rename(archivo_original_csv, archivo_nuevo_csv)
        if os.path.exists(archivo_original_png):
            os.rename(archivo_original_png, archivo_nuevo_png)

