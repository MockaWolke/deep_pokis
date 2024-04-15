import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cores", type=int, default=os.cpu_count())
parser.add_argument("--boxes_path", type=str, default="data/generated_boxes")
parser.add_argument("--imgs_path", type=str, default="data/generated")

args = parser.parse_args()

os.makedirs("data/generated_cropped", exist_ok=True)

CROP_FACTOR = 1.25

samples = pd.read_csv("data/test_generated_imgs.csv", index_col=0)

def save_crop(name):
    path = f"{args.imgs_path}/{name}.jpg"
    
    goal = f"data/generated_cropped/{name}.jpg"
    
    bbox_path = f"{args.boxes_path}/{name}.jpg.txt"
    
    
    if os.path.exists(goal):
        return False
    
    if not os.path.exists(bbox_path):
        return True
    

    with open(bbox_path, "r") as f:
        
        boxes = list(map(float, f.readlines()[0].split(" ")))


    _,x,y,w,h = boxes

    orig_img = Image.open(path)

    x,w = (x * orig_img.size[0], w * orig_img.size[0])
    
    y,h = (y * orig_img.size[1], h * orig_img.size[1])

    left = max(int(x - w/2*CROP_FACTOR),0)
    right = min(int(x + w/2*CROP_FACTOR), orig_img.size[0])
    bottom = max(int(y - h/2*CROP_FACTOR),0)
    top = min(int(y + h/2*CROP_FACTOR), orig_img.size[1])


    cropped_img = orig_img.crop((left,bottom, right, top))
    
    
    cropped_img.save(goal)
    
    return 0
    
    
    
ids = samples.Id.values.tolist()
    

def wrapper(image_id):
    try:
        save_crop(image_id)
    except Exception as e:
        print(e)


with Pool(args.cores) as pool:
        list(tqdm(pool.imap(wrapper, ids), total=len(ids)))
        