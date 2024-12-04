import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO("yolov8-hgnet.yaml")
   # model.load('yolov8n.pt') 
    model.train(data=r'ultralytics/cfg/datasets/FLIR_v2.yaml',
                cache=False,
                imgsz=640,
                epochs=2,
                batch=16,
                close_mosaic=0,
                workers=0,
                device= None,
                optimizer='SGD', # using SGD
                amp=False,# close amp
                )