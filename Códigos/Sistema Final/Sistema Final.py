import os
import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
import subprocess

subprocess.Popen(['lxterminal', '-e', 'python3', 'ventilador_vf.py'])

# === Parámetros de configuración ===
CONFIRMAR_CON_SENSOR = True      # Cambia a True si quieres usar el sensor de aire
PM25_UMBRAL = 100
PORT = "/dev/ttyUSB0"
BAUD = 9600
MODELO = "best.pt"
TIEMPO_ENTRE_CAPTURAS = 5       # segundos
TIEMPO_ENTRE_ALERTAS = 300       # 5 minutos = 300 segundos

IMG_TMP = "procesando.jpg"

def leer_pm25():
    try:
        with serial.Serial(PORT, BAUD, timeout=2) as ser:
            for _ in range(10):  # Intenta 10 lecturas como máximo
                data = ser.read(32)
                if len(data) != 32:
                    time.sleep(1)
                    continue
                if data[0] == 0x42 and data[1] == 0x4d:
                    pm25_standard = int.from_bytes(data[12:14], byteorder='big')
                    return pm25_standard
                time.sleep(1)
    except Exception as e:
        pass
    return None

def ejecutar_alerta():
    subprocess.Popen(['lxterminal', '-e', 'python3', 'alerta_vf.py'])

def rgb_to_pjf(R, G, B):
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

def obtener_rectangulos_color(img, mask, area_min=10):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= area_min:
            rects.append((x, y, w, h))
    return rects

def intersecta(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def fusionar_boxes(boxes):
    if not boxes:
        return []
    xs = [x for x, y, w, h in boxes]
    ys = [y for x, y, w, h in boxes]
    x2s = [x + w for x, y, w, h in boxes]
    y2s = [y + h for x, y, w, h in boxes]
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(x2s), max(y2s)
    return [(x_min, y_min, x_max - x_min, y_max - y_min)]

def procesar_imagen_desde_archivo(img_path, model, ultima_alarma):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return ultima_alarma
    img_rgb = img_bgr.copy()
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    res = model.predict(source=img_rgb, conf=0.01, save=False, save_txt=False)[0]
    
    boxes_combinadas = []
    for box in res.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if conf >= 0.25:
            boxes_combinadas.append((x1, y1, w, h))
        else:
            P, J, F = rgb_to_pjf(R, G, B)
            mask1 = P > 220
            mask2 = J > 60
            mask3 = F > 230
            fire_mask1 = mask1 & mask2
            fire_mask2 = mask2 & mask3
            fire_mask = fire_mask1 + fire_mask2

            R_split, G_split, B_split = cv2.split(img_rgb)
            RR, RG, RB = 190, 100, 140
            condicion_1 = (R_split > G_split) & (G_split > B_split)
            masc1 = img_rgb.copy()
            masc1[condicion_1 == 0] = 0
            condicion_2 = (masc1[:,:,0] > RR) & (masc1[:,:,1] > RG) & (masc1[:,:,2] < RB)
            masc2 = masc1.copy()
            masc2[condicion_2 == 0] = 0
            imagen_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = cv2.split(imagen_ycbcr)
            Y_mean = np.mean(Y)
            Cb_mean = np.mean(Cb)
            Cr_mean = np.mean(Cr)
            condicion_3 = (Y > Cb) & (Cr > Cb)
            masc3 = imagen_ycbcr.copy()
            masc3[condicion_3 == 0] = 0
            condicion_4 = (masc3[:,:,0] > Y_mean) & (masc3[:,:,1] > Cr_mean) & (masc3[:,:,2] < Cb_mean)
            masc4 = imagen_ycbcr.copy()
            masc4[condicion_4 == 0] = 0
            fire_mask_fy = condicion_2 * condicion_4
            fire_mask_fy = fire_mask_fy.astype(np.uint8) * 255
            fire_mask_f2 = fire_mask_fy & fire_mask
            fire_mask_f2 = fire_mask_f2.astype(np.uint8) * 255
            boxes_color = obtener_rectangulos_color(img_rgb, fire_mask_f2)
            for cb in boxes_color:
                if intersecta((x1, y1, w, h), cb):
                    area_cb = cb[2] * cb[3]
                    area_yolo = w * h
                    boxes_combinadas.append(cb if area_cb > area_yolo else (x1, y1, w, h))
                    break

    boxes_finales = fusionar_boxes(boxes_combinadas)

    img_rgb_dib = img_rgb.copy()
    for (x, y, w, h) in boxes_finales:
        cv2.rectangle(img_rgb_dib, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite("incendio.jpg", img_rgb_dib)
    
    hay_fuego = len(boxes_finales) > 0
    if hay_fuego:
        ahora = time.time()
        if (ahora - ultima_alarma) >= TIEMPO_ENTRE_ALERTAS:
            if CONFIRMAR_CON_SENSOR:
                pm25 = leer_pm25()
                if pm25 is not None and pm25 > PM25_UMBRAL:
                    ejecutar_alerta()
                    return ahora
            else:
                ejecutar_alerta()
                return ahora
    return ultima_alarma

if __name__ == "__main__":
    model = YOLO(MODELO)
    ultima_alarma = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        exit(1)

    first_frame = True
    while True:
        for _ in range(5):
            cap.read()
            time.sleep(0.05)
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(2)
            continue
        if first_frame:
            time.sleep(3)
            first_frame = False
        cv2.imwrite(IMG_TMP, frame)
        ultima_alarma = procesar_imagen_desde_archivo(IMG_TMP, model, ultima_alarma)
        time.sleep(TIEMPO_ENTRE_CAPTURAS)
    cap.release()
