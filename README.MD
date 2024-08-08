# Identify Object Position with Grid OpenCV & YoloV5

## Installation

```
git clone https://github.com/Aftaza/grid-position-yolov5.git
cd grid-position-yolov5
pip install -r requirements.txt
```

## Usage
`detect.py` uses for simple grid position with haarcascade_frontalface_default.  
```
python detect.py
# or
python oop-detect.py
```
`yolov5.py` uses for grid position with yolo_v5 model.  
```
# with CPU Process
python yolov5.py

# if you want run program with CUDA support
python yolov5.py cuda
```
> Note: You must export yolov5 model default .pt to .onnx format and list your clases in `/conf` folder.  

