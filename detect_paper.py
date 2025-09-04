# from ultralytics import YOLO

# # Load a COCO-pretrained YOLOv8n model
# model = YOLO("Desktop/research/YOLO_thermal_small/yolov8n.pt")

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLOv8n model on the 'bus.jpg' image
# # results = model("Desktop/research/Datasets/FLIR_YOLOv2_9/thermal/images/val/video-57kWWRyeqqHs3Byei-frame-000816-b6tuLjNco8MfoBs3d.jpg")
# # results = model("Desktop/research/YOLO_thermal_small/data")
# results = model.predict(
#     source="Desktop/research/YOLO_thermal_small/data/data_val",  # file/dir/URL/0(webcam)
#     save=True,                       # draw boxes and save images
#     save_txt=True,                   # (optional) label txt files
# )


#########################################
"""
Compare three YOLO models on the same images and place the results
side‑by‑side in a single picture.

Ultralytics YOLO ≥8.2 required.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# ─── 1.  Set paths ────────────────────────────────────────────────────────────
SOURCE_DIR  = Path("Desktop/research/YOLO_thermal_small/data")
OUTPUT_DIR  = Path("Desktop/research/YOLO_thermal_small/data_comparison_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)           # create if needed

MODEL_PATHS = {
    "yolov8n": "Desktop/research/YOLO_thermal_small/yolov8n.pt",
    "yolo_MS": "Desktop/research/YOLO_thermal_small/runs/detect/MobleNet4_slideloss_RGB23/weights/last.pt",
    "yolov_S": "Desktop/research/YOLO_thermal_small/runs/detect/MobleNet4_slideloss_RGB10/weights/last.pt",
}

# ─── 2.  Load models once (much faster than re‑loading per image) ─────────────
models = {name: YOLO(path) for name, path in MODEL_PATHS.items()}

# ─── 3.  Helper: make all images the same height before hconcat ───────────────
def resize_to_height(img, height: int):
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)

# ─── 4.  Iterate through every file in the source folder ──────────────────────
for img_path in sorted(SOURCE_DIR.iterdir()):
    if not img_path.is_file():
        continue                                          # skip sub‑folders
    canvases = []

    for name, model in models.items():
        # Run prediction on a single image; take the first (and only) result
        result   = model.predict(img_path, imgsz=640, conf=0.25, save=False)[0]
        rendered = result.plot()                    # BGR np.ndarray with boxes

        # Add model‑name banner for clarity
        cv2.putText(
            rendered, f"{name}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        canvases.append(rendered)

    # ─── 5.  Stitch the three renders side‑by‑side ────────────────────────────
    min_h         = min(im.shape[0] for im in canvases)
    canvases      = [resize_to_height(im, min_h) for im in canvases]
    comparison    = cv2.hconcat(canvases)               # single wide image

    # ─── 6.  Save with the same file‑name in the new folder ───────────────────
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), comparison)
    print(f"Saved {out_path}")

print("All comparisons finished")
