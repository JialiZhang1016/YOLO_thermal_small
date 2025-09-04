from ultralytics import YOLO

# Initialize your custom model
model = YOLO("runs/detect/MobleNet4_slideloss/weights/best.pt")

results = model.val(data="ultralytics/cfg/datasets/FLIR_v2_RGB_thermal.yaml", batch=128)

# success = model.export(format="onnx")