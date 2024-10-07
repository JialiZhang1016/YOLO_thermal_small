# Real-Time Object Detection with YOLO and Drones

This repository is dedicated to real-time object detection using YOLO. We leverage the **Ultralytics** YOLO framework to build a detection system using data collected from drones equipped with **RGB** and **thermal cameras**. Our goal is to develop a reliable detection system that can operate in real-time.

## Project Overview

-**Framework**: YOLO by Ultralytics

-**Data Collection**: Drone footage utilizing both RGB and thermal cameras

-**Platform**: MacOS, Windows, Linux

-**Primary Objective**: Real-time detection of selected objects using a dual-camera setup on drones

## Dataset

The dataset used in this project is the **FLIR dataset** (please find it in Teams). It includes RGB images and two types of thermal images:

-**RGB images**

-**Thermal** (color format)

-**Thermal_gray16** (16-bit grayscale format)

The dataset is organized into the following folder structure:

```

├── images

│   ├── test

│   ├── train

│   └── val

├── labels

│   ├── test

│   ├── train

│   └── val

├── test.txt

├── train.txt

└── val.txt

```

I have filtered the dataset to focus on 10 out of the original 15 classes, as listed below:

-`person`

-`bike`

-`car`

-`motor`

-`bus`

-`truck`

-`light`

-`hydrant`

-`sign`

-`other vehicle`

These categories were chosen for their relevance to our detection tasks. Other classes were excluded due to limited data availability, which could negatively impact training and detection quality.

### Data Preprocessing

To facilitate data conversion, I created a script named [coco2yolo.py](ultralytics/cfg/datasets/flir2yolo.py) that converts the FLIR dataset annotations to the YOLO format required by Ultralytics. The resulting structure aligns with YOLO requirements, and the labels are prepared for immediate use with the chosen categories.

Additionally, a configuration file, [FLIR_v2.yaml](ultralytics/cfg/datasets/FLIR_v2.yaml), is included to utilize thermal data specifically. This file can be easily modified to adapt to RGB or other data sources as needed.

## Getting Started

To set up the project and prepare the data for training, follow these steps:

1.**Clone the Repository**

```bash

git clone https://github.com/JialiZhang1016/Ultralytics.git

cd Ultralytics

```

2.**Create envirorment for this repo**

- Follow the [official guide](README_offical.md) to set up the environment.

3.**Data Preparation**

- Ensure the FLIR thermal dataset is structured as shown above.
- If needed, run [coco2yolo.py](ultralytics/cfg/datasets/flir2yolo.py) to convert the annotations if you are using the original COCO format.

5.**Configuration**

- Edit the paths in the [FLIR_v2.yaml](ultralytics/cfg/datasets/FLIR_v2.yaml) file to match your dataset location.

6.**Training**

    ```bash

yolo train data=FLIR_v2.yaml

    ```

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to Ultrilytics for their YOLO framework and the creators of the FLIR dataset for providing valuable data for object detection tasks.
