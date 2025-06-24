# FUNCIONAMIENTO DEL VENTILADOR OPTIMIZADO

import time
import RPi.GPIO as GPIO

# ==== CONFIGURACIÓN ====
GPIO_PIN = 17       # Pin GPIO a usar (modo BCM)
UMBRAL_TEMP = 60    # Umbral de temperatura en °C
INTERVALO = 5       # Intervalo de lectura en segundos

# ==== INICIALIZACIÓN ====
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)
GPIO.output(GPIO_PIN, GPIO.LOW)  # Ventilador apagado al inicio

def obtener_temperatura():
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        return float(f.readline()) / 1000.0

try:
    while True:
        temp = obtener_temperatura()
        GPIO.output(GPIO_PIN, GPIO.HIGH if temp > UMBRAL_TEMP else GPIO.LOW)
        time.sleep(INTERVALO)
except KeyboardInterrupt:
    pass
finally:
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO.cleanup()

