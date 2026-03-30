# Download Model Weights

Model weight files are excluded from GitHub (too large).
Download them and place in this `models/` folder.

## Required Files

### 1. yolov3-tiny.weights (~35 MB)
```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://pjreddie.com/media/files/yolov3-tiny.weights',
    'models/yolov3-tiny.weights'
)
print('Done!')
"
```
Or open in browser: https://pjreddie.com/media/files/yolov3-tiny.weights

### 2. yolov3-tiny.cfg (auto-downloaded by app.py on first run)

## Files Already Included
- `deploy.prototxt` — MobileNet config
- `coco.names` — 80 COCO object class labels

## After Download, models/ should contain:
```
models/
├── yolov3-tiny.weights   ← download this (~35 MB)
├── yolov3-tiny.cfg       ← auto-downloaded
├── coco.names            ← included ✓
└── deploy.prototxt       ← included ✓
```
