from ultralytics import YOLO
import csv

model = YOLO("/home/tsw96d/YOLOComparision/YOLO_thermal_small/MobleNet43/weights/best.pt")

results = model.val(data="/home/tsw96d/YOLOComparision/FLIR_v2.yaml", project="../../outputs/small", name = "YOLO11", save_json=True)
