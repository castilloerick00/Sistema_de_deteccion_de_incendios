# LECTURA DE .TXT E IMPRESION DE METRICAS ACC, TPR Y TNR

import numpy as np


def cargar_resultados_con_etiquetas(archivo):
    """
    Lee un archivo de texto con líneas de resultados y devuelve diccionarios de valores reales y predichos.
    Intenta decodificar en UTF-8 y, si falla, usa Latin-1.
    Formato esperado de línea: "<nombre_img>: <resultado> (Detección...)".
    """
    reales = {}
    predichos = {}

    # Leer contenido raw
    with open(archivo, 'rb') as f_bin:
        data = f_bin.read()

    # Intentar decodificar
    try:
        contenido = data.decode('utf-8')
    except UnicodeDecodeError:
        print(f"Advertencia: no se pudo decodificar con UTF-8. Probando Latin-1...")
        contenido = data.decode('latin-1')

    # Procesar línea a línea
    for linea in contenido.splitlines():
        linea = linea.strip()
        if not linea or ':' not in linea:
            continue
        nombre_img, resto = linea.split(':', 1)
        nombre_img = nombre_img.strip()
        resto = resto.strip()

        # Filtrar solo líneas con "Detección"
        if 'detección' not in resto.lower():
            continue

        # Extraer predicción: parte antes del paréntesis
        pred_texto = resto.split('(')[0].strip().lower()
        pred = 'sí' in pred_texto or 'si' in pred_texto

        # Marcas de correcta/errónea
        correcta = 'correcta' in resto.lower()

        # Etiqueta real según si fue correcta
        real = pred if correcta else not pred

        reales[nombre_img] = real
        predichos[nombre_img] = pred

    return reales, predichos


def calcular_metricas(reales, predichos):
    """Calcula TP, TN, FP, FN y métricas básicas."""
    TP = FP = TN = FN = 0
    for img, real in reales.items():
        pred = predichos.get(img, False)
        if real and pred:
            TP += 1
        elif not real and not pred:
            TN += 1
        elif not real and pred:
            FP += 1
        else:
            FN += 1

    total = TP + TN + FP + FN
    if total == 0:
        raise ValueError("No se encontraron resultados válidos. Revisa la codificación o el formato del archivo.")

    accuracy = (TP + TN) / total
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0

    return TP, FP, TN, FN, accuracy, tpr, tnr


if __name__ == '__main__':
    ruta_txt = '/home/pi/TESIS/Codigos/Imagenes/Resultados/Escenario3/Color/condicion1/deteccion_fuego.txt'
    #ruta_txt = '/home/pi/TESIS/Codigos/Imagenes/Resultados/IA/Escenario1/HAAR/deteccion_fuego.txt'
    try:
        reales, predicciones = cargar_resultados_con_etiquetas(ruta_txt)
        print(f"Líneas procesadas: {len(reales)}")
        TP, FP, TN, FN, accuracy, tpr, tnr = calcular_metricas(reales, predicciones)

        print("\n=== MATRIZ DE CONFUSIÓN ===")
        print(f"Verdaderos Positivos (TP): {TP}")
        print(f"Falsos Positivos (FP): {FP}")
        print(f"Verdaderos Negativos (TN): {TN}")
        print(f"Falsos Negativos (FN): {FN}\n")

        print("=== MÉTRICAS ===")
        print(f"Accuracy (ACC): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"True Positive Rate (TPR/Recall): {tpr:.4f} ({tpr*100:.2f}%)")
        print(f"True Negative Rate (TNR/Specificity): {tnr:.4f} ({tnr*100:.2f}%)")

        print("\n=== CASOS DE ERROR ===")
        for img, real in reales.items():
            pred = predicciones[img]
            if real != pred:
                r_text = 'Sí' if real else 'No'
                p_text = 'Sí' if pred else 'No'
                print(f"Imagen: {img} | Real: {r_text} | Predicción: {p_text}")

    except Exception as e:
        print(f"Error: {e}")

