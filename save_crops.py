from ultralytics import YOLO
import os
import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob


CROP_FACTOR = 2.0

model = YOLO('best.pt')  



def save_crop(path, name):
    save_dir = f"data/{name}_cropped"
    os.makedirs(save_dir, exist_ok=True)
    
    pred = model.predict(path, verbose = False)[0]

    boxes = pred.boxes.numpy()
    if len(boxes.cls) == 0:
        return
    
    best_arg = np.argsort(boxes.cls, 0)[0]
    x,y,w,h = boxes.xywh[best_arg]

    orig_img = Image.open(pred.path)

    left = max(int(x - w/2*CROP_FACTOR),0)
    right = min(int(x + w/2*CROP_FACTOR), orig_img.size[0])
    bottom = max(int(y - h/2*CROP_FACTOR),0)
    top = min(int(y + h/2*CROP_FACTOR), orig_img.size[1])


    cropped_img = orig_img.crop((left,bottom, right, top))
    
    
    cropped_img.save(os.path.join(save_dir, os.path.basename(path)))
    
    
for path in tqdm.tqdm(glob("data/train/*.png")):
    
    save_crop(path, "train")
    
for path in tqdm.tqdm(glob("data/test/*.png")):
    
    save_crop(path, "test")
    
    