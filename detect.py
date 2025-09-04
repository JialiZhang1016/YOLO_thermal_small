import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained model
    model = YOLO("yolov8_200_64_best.pt")  # Change to your trained model path
    
    # Perform detection on images in the data folder
    results = model.predict(
        source="data",  # Path to your data folder
        conf=0.25,      # Confidence threshold
        iou=0.45,       # NMS IoU threshold
        imgsz=640,      # Image size
        save=True       # Save results
    )