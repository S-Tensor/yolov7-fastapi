from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import base64
import io
from PIL import Image
import torch
import numpy as np
import sys
import os
import uvicorn
# Add YOLOv7 repo to path
sys.path.append("yolov7")
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox

app = FastAPI()

# Load YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = attempt_load(r"C:\Users\53581\PycharmProjects\yolo_test1\yolov7\yolov7.pt", map_location=device)
model.eval()
model.to(device)

# Load class names (COCO dataset)
CLASSES = model.names


class ImageInput(BaseModel):
    image_base64: str


@app.post("/detect")
async def detect_objects(input_data: ImageInput):
    # Decode base64 to image
    try:

        image_data = base64.b64decode(input_data.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return {"error": "Invalid base64 image"}

    # Preprocess
    img = np.array(image)
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    results = []

    # Extract detected classes
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                label = CLASSES[int(cls)]
                results.append(label)

    return {"detected_objects": results}
