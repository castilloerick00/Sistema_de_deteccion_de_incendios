# 🔥 Sistema Inteligente de Detección y Alerta Temprana de Incendios Forestales

Este repositorio contiene el desarrollo completo del proyecto de integración curricular titulado **“Sistema de detección de incendios basado en tecnología IoT y procesamiento de imágenes”**, implementado en Python sobre hardware de bajo costo.

El sistema fue motivado por la creciente incidencia de incendios en Ecuador durante períodos de sequía (como en 2024), y está diseñado para operar de manera autónoma en zonas rurales o de difícil acceso, integrando visión por computadora, sensores ambientales y comunicación 4G.

---

## 📁 Estructura del repositorio

```
├── 📂 Códigos  
│   ├── 📂 Algoritmos  
│   │   ├── 🐍 Método 1.py  
│   │   ├── 🐍 Método 2.py  
│   │   ├── 🐍 Método 3.py  
│   │   ├── 🐍 Método 4.py  
│   │   ├── 🐍 Método 5.py  
│   │   ├── 🐍 Método 6.py  
│   │   ├── 🐍 Método 7.py  
│   │   ├── 🐍 Método 11.py  
│   │   ├── 🐍 Método 12.py  
│   │   ├── 🐍 Método 13.py 
│   │   ├── 🐍 Método 14.py   
│   │   ├── 📂 Método 8
│   │   ├── 📂 Método 9 
│   │   └── 📂 Método 10
│   ├── 📂 Extras  
│   │   ├── 📂 Ajuste de imágenes  
│   │   │   ├── 🐍 Redimencionar.py  
│   │   │   └── 🐍 Renombrar.py  
│   │   └── 📂 Análisis de métricas  
│   │       ├── 🐍 Métricas de rendimiento.py  
│   │       └── 🐍 Métricas estadísticas.py  
│   └── 📂 Sistema Final  
│       ├── 📦 best.pt  
│       ├── 📦 best25_1.pt  
│       ├── 🐍 Control Ventilador.py  
│       ├── 🐍 Envio Alerta.py  
│       └── 🐍 Sistema Final.py  
├── 📂 Datasets  
│   └── 📄 datasets.txt  
└── 📂 Resultados  
    ├── 📂 Métricas Estadísticas  
    │   ├── 📂 Escenario 1  
    │   │   ├── 📄 Método 1.txt  
    │   │   ├── 📄 Método 2.txt  
    │   │   ├── 📄 Método 3.txt  
    │   │   ├── 📄 Método 4.txt  
    │   │   ├── 📄 Método 5.txt  
    │   │   ├── 📄 Método 6.txt  
    │   │   ├── 📄 Método 7.txt  
    │   │   ├── 📄 Método 8.txt  
    │   │   ├── 📄 Método 9.txt  
    │   │   ├── 📄 Método 10.txt  
    │   │   ├── 📄 Método 11.txt  
    │   │   ├── 📄 Método 12.txt  
    │   │   ├── 📄 Método 13.txt  
    │   │   └── 📄 Método 14.txt  
    │   ├── 📂 Escenario 2  
    │   │   ├── 📄 Método 1.txt  
    │   │   ├── 📄 Método 2.txt  
    │   │   ├── 📄 Método 3.txt  
    │   │   ├── 📄 Método 4.txt  
    │   │   ├── 📄 Método 5.txt  
    │   │   ├── 📄 Método 6.txt  
    │   │   ├── 📄 Método 7.txt  
    │   │   ├── 📄 Método 8.txt  
    │   │   ├── 📄 Método 9.txt  
    │   │   ├── 📄 Método 10.txt  
    │   │   ├── 📄 Método 11.txt  
    │   │   ├── 📄 Método 12.txt  
    │   │   ├── 📄 Método 13.txt  
    │   │   └── 📄 Método 14.txt  
    │   └── 📂 Escenario 3  
    │       ├── 📂 Condición 1  
    │       │   ├── 📄 Método 9.txt  
    │       │   ├── 📄 Método 12.txt  
    │       │   └── 📄 Sistema Final.txt  
    │       ├── 📂 Condición 2  
    │       │   ├── 📄 Método 9.txt  
    │       │   ├── 📄 Método 12.txt  
    │       │   └── 📄 Sistema Final.txt  
    │       └── 📂 Condición 3  
    │           ├── 📄 Método 9.txt  
    │           ├── 📄 Método 12.txt  
    │           └── 📄 Sistema Final.txt  
    └── 📂 Métricas Rendimiento  
        ├── 📊 Método 1.csv  
        ├── 📊 Método 2.csv  
        ├── 📊 Método 3.csv  
        ├── 📊 Método 4.csv  
        ├── 📊 Método 5.csv  
        ├── 📊 Método 6.csv  
        ├── 📊 Método 7.csv  
        ├── 📊 Método 8.csv  
        ├── 📊 Método 9.csv  
        ├── 📊 Método 10.csv  
        ├── 📊 Método 11.csv  
        ├── 📊 Método 12.csv  
        ├── 📊 Método 13.csv  
        ├── 📊 Método 14.csv  
        ├── 📊 Sistema Final-Con ventilación.csv  
        └── 📊 Sistema Final-Sin ventilación.csv  
```




---

## 🧠 Algoritmos del Sistema Final

- **Método 9 (YOLOv8)**: Detección basada en inteligencia artificial con entrenamiento personalizado.
- **Método 12**: Detección basada en espacios de color (PJF, RGB, YCbCr).
- **Sistema Final**: Algoritmo híbrido que combina los anteriores mediante lógica jerárquica.

---

## 🧪 Escenarios de Validación para el Escenario Final

El sistema fue validado en tres condiciones representativas del entorno:

1. **Alta iluminación**: Simula días despejados en temporada de sequía.
2. **Interferencia visual**: Introducción de elementos similares al fuego.
3. **Escenario estándar**: Condiciones típicas de la ciudad de Cuenca.

---

## 📊 Resultados Relevantes

| Condición             | ACC (%) | TPR (%) | TNR (%) |
|----------------------|---------|---------|---------|
| Alta iluminación     | 99.82   | 99.64   | 100.00  |
| Interferencia visual | 96.55   | 99.73   | 93.45   |
| Escenario estándar   | 98.50   | 98.91   | 98.09   |

- Sensor PMS5003: validación por partículas PM2.5 > 100 µg/m³  
- Consumo energético: 519.9 mA  
- Uso CPU: 38.3 %  
- Temperatura: < 39.3 °C (con ventilación automática)

---

## ⚙️ Tecnologías y Librerías

- Python 3.x  
- OpenCV, NumPy, Matplotlib  
- Ultralytics YOLOv8  
- Raspberry Pi  
- PMS5003, módulo GPS y red 4G  
- Protocolos de comunicación serial y PPP  

---

## 🚀 Ejecución del sistema

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/tesis-deteccion-incendios.git

---

## 🧾 Créditos

📚 Proyecto de Integración Curricular – Facultad de Ingeniería  
🏛️ Universidad de Cuenca  
📡 Carrera de Ingeniería en Telecomunicaciones  
👨‍💻 Autores: Erick Alexander Castillo Matamoros - Robert Sebastián Chalco Montalván  
🧑‍🏫 Tutor: Ing. Santiago Renán González Martínez, Ph.D.  
📅 Año: 2025


---
