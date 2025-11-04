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

# Crear carpeta para capturas
os.makedirs("capturas", exist_ok=True)

# Tiempo de inicio
start_time = time.time()

# Función para generar color único por clase
def get_color(label):
    hash_val = int(hashlib.md5(label.encode()).hexdigest(), 16)
    r = (hash_val >> 16) & 255
    g = (hash_val >> 8) & 255
    b = hash_val & 255
    return (r, g, b)

# Cargar modelo YOLOv8
model = YOLO("yolov8n.pt")

# Región de captura
monitor = {
    "top": 160,
    "left": 0,
    "width": 960,
    "height": 540}

#Video
video_writer = cv2.VideoWriter(
    "demo_deteccion.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    20.0,
    (monitor["width"], monitor["height"])
)

# Contadores
total_counter = Counter()
log = []
image_count = 0
last_second = int(time.time())

with mss.mss() as sct:
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model(frame, verbose=False)[0]
        frame_counter = Counter()
        current_second = int(time.time())
        person_count = 0

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label != "person":
                    continue
                person_count += 1
                frame_counter[label] += 1
                total_counter[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = get_color(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Registro por segundo
        if current_second != last_second:
            fecha_hora = datetime.fromtimestamp(current_second).strftime("%Y-%m-%d %H:%M:%S")
            log.append([current_second, fecha_hora, person_count])
            last_second = current_second

            # Guardar imagen solo si hay al menos una persona
            if person_count > 0:
                filename = f"capturas/frame_{current_second}_{person_count}p.jpg"
                cv2.imwrite(filename, frame)
                image_count += 1

        # Calcular tiempo transcurrido
        elapsed_time = int(time.time() - start_time)
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Panel de conteo
        labels = sorted(total_counter.keys())
        num_labels = len(labels)
        cols = 3
        rows = (num_labels + cols - 1) // cols
        col_width = 200
        row_height = 25
        panel_height = max(100, 30 + (rows + 2) * row_height)
        panel_width = col_width * cols
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

        cv2.putText(panel, "Conteo de objetos", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Tiempo activo: {formatted_time}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(panel, f"Imagenes guardadas: {image_count}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)


        for i, label in enumerate(labels):
            current = frame_counter.get(label, 0)
            total = total_counter[label]
            text = f"{label}: {current} ({total})"
            col = i % cols
            row = i // cols
            x = 10 + col * col_width
            y = 70 + row * row_height
            cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(label), 2)

        video_writer.write(frame)
        
        cv2.imshow("Detección", frame)
        cv2.imshow("Panel de conteo", panel)

        if cv2.waitKey(1) == 27:
            break

# Guardar log como CSV
with open("person_count_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp","datetime", "person_count"])
    writer.writerows(log)

# Calculo de promedio y estimación por hora
total_personas = sum(row[1] for row in log)
total_segundos = len(log)
promedio_por_segundo = total_personas / total_segundos if total_segundos > 0 else 0
estimado_por_hora = promedio_por_segundo * 3600

# Mostrar resultados finales
print(f"Promedio por segundo: {promedio_por_segundo:.2f}")
print(f"Estimación por hora: {estimado_por_hora:.0f} personas")

# Mostrar tiempo total de ejecución
total_runtime = int(time.time() - start_time)
h = total_runtime // 3600
m = (total_runtime % 3600) // 60
s = total_runtime % 60
print(f"Tiempo total de ejecución: {h:02}:{m:02}:{s:02}")

cv2.destroyAllWindows()

video_writer.release()
