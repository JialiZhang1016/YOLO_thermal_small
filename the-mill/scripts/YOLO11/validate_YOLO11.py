from ultralytics import YOLO
import csv

model = YOLO("/home/tsw96d/YOLOComparision/YOLO_thermal_small/the-mill/outputs/nano/training/YOLOV11/weights/best.pt")

results = model.val(data="/home/tsw96d/YOLOComparision/YOLO_thermal_small/the-mill/scripts/FLIR_v2.yaml", project="../../outputs/nano", name = "YOLO11", save_json=True)

map_values = results
header = ['Class', 'mAP@0.5']