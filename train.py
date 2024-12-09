import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO("yolov8-hgnet.yaml")
   # model.load('yolov8n.pt') 
    model.train(data=r'/home/tsw96d/YOLOComparision/FLIR_v2.yaml',
                cache=True,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=0,
                workers=8,
                device= None,
                optimizer='SGD', # using SGD
                amp=True,# close amp
                )