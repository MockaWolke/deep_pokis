from ultralytics import YOLO
import os
import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import pandas as pd

CROP_FACTOR = 2.0

model = YOLO('last.pt')  

samples = pd.read_csv("data/generated_imgs.csv", index_col=0)

def save_crop(name):
    path = f"data/generated/{name}.jpg"
    
    goal = f"data/generated_cropped/{name}.jpg"
    
    if os.path.exists(goal):
        return
    
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
    
    
    cropped_img.save(goal)
    
    
for i in tqdm.tqdm(samples.Id):
    save_crop(i)