from ultralytics import YOLO
import csv

model = YOLO("runs/detect/MobleNet4_slideloss/weights/best.pt")

results = model.val(data="/home/tsw96d/YOLOComparision/FLIR_v2.yaml",batch=128, device=[0,1])
