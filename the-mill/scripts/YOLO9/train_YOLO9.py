from ultralytics import YOLO


model = YOLO("YOLO9n.yaml")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/home/tsw96d/YOLOComparision/YOLO_thermal_small/the-mill/scripts/FLIR_v2.yaml", epochs=200, batch=64, project="../../outputs/nano/training", name = "YOLO9") 

# Evaluate the model's performance on the validation set
results = model.val(data="/home/tsw96d/YOLOComparision/YOLO_thermal_small/the-mill/scripts/FLIR_v2.yaml", epochs=200, batch=64, project="../../outputs/nano/validation", name = "YOLO9")

# Export the model to ONNX format
success = model.export(format="onnx")