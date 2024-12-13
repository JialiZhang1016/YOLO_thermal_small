from ultralytics import YOLO

# Initialize your custom model
model = YOLO("./ultralytics/cfg/models/v8/yolov8_MobileNetv4.yaml")

results = model.train(data="/home/tsw96d/YOLOComparision/FLIR_v2.yaml", epochs=200, batch=64, project="../outputs/training", name = "MobleNet4") 
