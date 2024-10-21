from ultralytics import YOLO


model = YOLO("yolov8n.pt")

results = model.train(data="./FLIR_v2.yaml", epochs=200, project="training", name="YOLOV8", batch=64)

# Evaluate the model's performance on the validation set
results = model.val(project="validation", name="YOLOV8")

# Export the model to ONNX format
success = model.export(format="onnx")