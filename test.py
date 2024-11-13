from ultralytics import YOLO

# Initialize your custom model
model = YOLO("./ultralytics/cfg/models/v8/yolov8_MobileNetv4.yaml")

# Train the model using COCO128
model.train(data="coco.yaml", epochs=10, imgsz=640, batch=16)

# model.val(data="coco128.yaml")
