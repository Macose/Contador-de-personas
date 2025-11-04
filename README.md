# Detección de Personas en Tiempo Real con YOLOv8

Este proyecto utiliza Python y YOLOv8 para detectar personas en tiempo real desde una región específica de la pantalla. Captura imágenes relevantes, genera estadísticas por segundo y por hora, y graba el proceso en video. Ideal para aplicaciones de análisis de tráfico, seguridad visual o benchmarking de actividad humana.

## Características
- Detección en tiempo real con YOLOv8 (`ultralytics`)
- Captura automática de imágenes si se detectan personas
- Panel visual con conteo por clase y tiempo activo
- Registro en CSV con timestamp y fecha legible
- Grabación de video del proceso (`demo_deteccion.avi`)
- Código modular y fácilmente adaptable

## Dependencias principales

Estas son las bibliotecas utilizadas en el script:

```python
import cv2
import numpy as np
import mss
from ultralytics import YOLO
from collections import Counter
import hashlib
import time
import csv
import os
from datetime import datetime
