from ultralytics import YOLO


model = YOLO("yolov10n.yaml")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="./FLIR_v2.yaml", epochs=200, project="training", name="YOLOV10", batch=64)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format="onnx")