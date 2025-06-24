# sudo lsof /dev/ttyS0
# sudo kill -9 3590

#!/usr/bin/python3
# -*- coding:utf-8 -*-

import RPi.GPIO as GPIO
import serial
import time
import requests
import os
import subprocess

power_key = 6
ser = serial.Serial('/dev/ttyS0', 115200)
ser.flushInput()
IMG_ORIGINAL = 'incendio.jpg'
BOT_TOKEN = '7334352246:AAF1_MrhFPpv-6xhDs8jSnQwClrHaI-m920'
GROUP_CHAT_ID = '-1002746077480'
MAX_INTENTOS_GPS = 30
DELAY_GPS_PRIMERA_LECTURA = 10
DELAY_GPS_INTENTOS = 3

def power_on(power_key):
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(power_key, GPIO.OUT)
    time.sleep(0.1)
    GPIO.output(power_key, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(power_key, GPIO.LOW)
    time.sleep(20)
    ser.flushInput()

def power_down(power_key):
    GPIO.output(power_key, GPIO.HIGH)
    time.sleep(3)
    GPIO.output(power_key, GPIO.LOW)
    time.sleep(18)

def send_at(command, back, timeout):
    ser.write((command+'\r\n').encode())
    time.sleep(timeout)
    if ser.inWaiting():
        time.sleep(0.01)
        rec_buff = ser.read(ser.inWaiting())
        if rec_buff != b'' and back in rec_buff.decode(errors='ignore'):
            s = rec_buff.decode(errors='ignore').replace('\n','').replace('\r','').replace('AT','').replace('+CGPSINFO','').replace(': ','')
            try:
                Lat = s[:2]
                SmallLat = s[2:11]
                NorthOrSouth = s[12]
                Long = s[14:17]
                SmallLong = s[17:26]
                EastOrWest = s[27]
                FinalLat = float(Lat) + (float(SmallLat)/60)
                FinalLong = float(Long) + (float(SmallLong)/60)
                if NorthOrSouth == 'S': FinalLat = -FinalLat
                if EastOrWest == 'W': FinalLong = -FinalLong
                return (FinalLat, FinalLong)
            except:
                return 0
    return 0

def get_gps_position(max_intentos=30):
    send_at('AT+CGPS=1,1','OK',1)
    time.sleep(DELAY_GPS_PRIMERA_LECTURA)
    intentos = 0
    while True:
        intentos += 1
        res = send_at('AT+CGPSINFO','+CGPSINFO: ',1)
        if isinstance(res, tuple):
            send_at('AT+CGPS=0','OK',1)
            return res
        if max_intentos and intentos >= max_intentos:
            send_at('AT+CGPS=0','OK',1)
            return None
        time.sleep(DELAY_GPS_INTENTOS)

def activar_4g():
    subprocess.Popen(['lxterminal', '-e', 'sudo pon sim7600'])
    time.sleep(5)

def enviar_alerta_telegram(lat, lon, imagen):
    MENSAJE = (
        "🚨 *ALERTA DE INCENDIO FORESTAL DETECTADA AUTOMÁTICAMENTE*\n\n"
        "El sistema inteligente ha identificado un posible incendio en la siguiente ubicación:\n"
        f"\n📍 *Coordenadas:* {lat}, {lon}\n"
        f"🌐 [Ver ubicación en Google Maps](https://maps.google.com/?q={lat},{lon})\n\n"
        "Por favor, verificar la imagen adjunta y activar el protocolo correspondiente.\n"
        "\n_Mensaje generado automáticamente por el sistema IoT de monitoreo de incendios._"
    )
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    with open(imagen, 'rb') as image_file:
        files = {'photo': image_file}
        data = {
            'chat_id': GROUP_CHAT_ID,
            'caption': MENSAJE,
            'parse_mode': 'Markdown'
        }
        requests.post(url, files=files, data=data)

if __name__ == "__main__":
    try:
        power_on(power_key)
        coords = get_gps_position(max_intentos=MAX_INTENTOS_GPS)
        if coords and os.path.exists(IMG_ORIGINAL):
            lat, lon = coords
            activar_4g()
            enviar_alerta_telegram(lat, lon, IMG_ORIGINAL)
        power_down(power_key)
    except:
        pass
    finally:
        GPIO.cleanup()
