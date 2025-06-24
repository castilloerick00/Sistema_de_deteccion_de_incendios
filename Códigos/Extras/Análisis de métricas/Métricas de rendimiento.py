# LECTURA Y ALMACENAMIENTO DE METRICAS DE CONSUMO ENERGETICO Y DE HARDWARE

#!/usr/bin/env python3

from ina219 import INA219, DeviceRangeError
import psutil
import matplotlib.pyplot as plt
import time
import csv
import signal
import sys

# Configuración del INA219
SHUNT_OHMS = 0.1
MAX_EXPECTED_AMPS = 3
I2C_BUS = 1
INA_ADDRESS = 0x40

INTERVAL = 1  # segundos
tiempo_transcurrido = []
currents = []
temperatures = []
cpu_usage = []
ram_usage = []
start_time = time.time()
stop_requested = False

# Señal de parada limpia (cuando otro proceso lo termina)
def handle_stop_signal(signum, frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, handle_stop_signal)
signal.signal(signal.SIGINT, handle_stop_signal)

ina = INA219(SHUNT_OHMS, MAX_EXPECTED_AMPS, address=INA_ADDRESS, busnum=I2C_BUS)
ina.configure(ina.RANGE_16V, ina.GAIN_AUTO)

def get_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read()) / 1000.0
    except:
        return None

print("Iniciando monitoreo...")

try:
    while not stop_requested:
        elapsed = int(time.time() - start_time)
        tiempo_transcurrido.append(elapsed)

        try:
            current_ma = ina.current()  # corriente en miliamperios
            currents.append(current_ma)
        except DeviceRangeError:
            currents.append(None)

        cpu_usage.append(psutil.cpu_percent(interval=0.5))
        ram_usage.append(psutil.virtual_memory().percent)
        temperatures.append(get_temp())

        # time.sleep(INTERVAL - 0.5)  # ya se considera 0.5 en cpu_percent

finally:
    print("Monitoreo finalizado. Guardando CSV y generando gráfico...")

    with open('/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tiempo', 'CPU', 'RAM', 'Temp', 'Corriente (mA)'])
        for i in range(len(tiempo_transcurrido)):
            writer.writerow([
                tiempo_transcurrido[i],
                cpu_usage[i],
                ram_usage[i],
                temperatures[i] if temperatures[i] is not None else '',
                currents[i] if currents[i] is not None else ''
            ])

    def get_x_ticks(time_list, num_labels=10):
        total_time = time_list[-1]
        step = total_time // (num_labels - 1) if num_labels > 1 else 1
        return list(range(0, total_time + 1, step))

    x_ticks = get_x_ticks(tiempo_transcurrido)

    plt.figure(figsize=(14, 6))
    plt.plot(tiempo_transcurrido, cpu_usage, label="CPU (%)", color='purple')
    plt.plot(tiempo_transcurrido, ram_usage, label="RAM (%)", color='brown')
    if any(t is not None for t in temperatures):
        plt.plot(tiempo_transcurrido, temperatures, label="Temp CPU (°C)", color='red')
    if any(c is not None for c in currents):
        plt.plot(tiempo_transcurrido, currents, label="Corriente (mA)", color='orange')

    plt.title("Uso del sistema y corriente consumida")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Uso (%) / Temp (°C) / Corriente (mA)")
    plt.xticks(ticks=x_ticks)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/pi/TESIS/Codigos/Metricas/Barras_y_Lineal/monitoreo.png")
    plt.close()

