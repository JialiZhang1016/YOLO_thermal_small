from ultralytics import YOLO

# Initialize your custom model
model = YOLO("./ultralytics/cfg/models/v8/yolov8n.yaml") # Mobile+loss+CIB
# model = YOLO("./ultralytics/cfg/models/v8/yolov8_MobileNetv4.yaml") # Mobile+loss+CIB
# model = YOLO("./ultralytics/cfg/models/v8/yolov8_MobileNetv4_clean.yaml") # Mobile+loss
# model = YOLO("./ultralytics/cfg/models/v8/yolov8n.yaml") # loss
# model = YOLO("./ultralytics/cfg/models/v8/yolov8_MobileNetv4.yaml") # Mobile


results = model.train(data="ultralytics/cfg/datasets/FLIR_v2.yaml", epochs=200, batch=128, imgsz=640, name = "MobleNet4_slideloss_RGB")

# FLIR_v2_RGB.yaml
# FLIR_v2.yaml
# FLIR_v2_RGB_thermal.yaml