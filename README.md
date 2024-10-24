# Military Objects detection on video stream using OpenCV

Script based on OpenCV and YOLO with custom datasets for Russian Ground Vehicles and a variety of objects from YOLO and YOLO-WORLD datasets.

## Installation

Only requires Python3, other dependencies and models handled on fly.

## Usage
Before launching script could be configured to use civil or military mode as well as specify sizes of models to use.
Search mode which could be also configured in script for detecting particular thing on the video stream covers both civil and military objects but drastically lacks in accuracy, range and performance.

Unix
```shell
chmod +x run.sh
./run.sh
```
Windows
```shell
# On Windows install dependencies from requirements.txt and python3.12-venv before running
python3 ./src/main.py
```
