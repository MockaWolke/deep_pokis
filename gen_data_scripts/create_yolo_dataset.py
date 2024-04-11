import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import hashlib
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import random
random.seed(0)
np.random.seed(0)

Y_SIZE = 256
X_SIZE = 256
CORES = 8
TIMES = 4


df = pd.read_csv("data/parsed_from_index.csv", index_col=0)

background_imgs = glob("data/places_imgs/*/*/*.jpg")
random.shuffle(background_imgs)

val_split = int(len(background_imgs) * 0.2)
val_imgs = background_imgs[:val_split]
train_imgs = background_imgs[val_split:]


def turn_tranparent(img):
    img = img.convert("RGBA")

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img


def create_img(path):
    img = Image.open(path).resize((Y_SIZE, X_SIZE)).convert("RGBA")
    back_ground = Image.new("RGBA", img.size)

    poke_id = np.random.randint(0, len(df.image_path))
    pokemon = Image.open(df.image_path.iloc[poke_id])
    pokemon = turn_tranparent(pokemon)

    if np.random.randint(0,2) > 0:
        pokemon = pokemon.transpose(Image.FLIP_LEFT_RIGHT)

    y_center = np.random.uniform(0.15, 0.85)
    x_center = np.random.uniform(0.15, 0.85)

    width_factor = np.random.uniform(0.07, 0.20)

    resize_y = int(1 / (pokemon.size[0] / Y_SIZE / width_factor) * pokemon.size[0])
    resize_x = int(1 / (pokemon.size[1] / Y_SIZE / width_factor) * pokemon.size[1])

    pokemon = pokemon.resize((resize_y, resize_x))
    y_width, x_width = pokemon.size

    upper_corner = int(y_center * Y_SIZE - (y_width/2))
    left_corner = int(x_center * X_SIZE - (x_width/2))
    back_ground.paste(pokemon, (upper_corner, left_corner), pokemon.split()[3])
    img = Image.alpha_composite(img, back_ground).convert("RGB")

    yolo_format = f"0 {(y_center)} {(x_center)} {y_width/Y_SIZE} {x_width/X_SIZE}"
    
    row = df.iloc[poke_id]
    
    file_name = f"{row.primary}_{row.secondary}_{hashlib.md5(img.tobytes()).hexdigest()}"
    
    return img, yolo_format, file_name


def process_image_train(image_path):
    name = "train"
    os.makedirs(f"data/yolo/images/{name}",exist_ok=True)
    os.makedirs(f"data/yolo/labels/{name}",exist_ok=True)
    img, yolo_format, file_name = create_img(image_path)
    img.save(os.path.join(f"data/yolo/images/{name}", file_name + ".jpg"))
    
    with open(os.path.join(f"data/yolo/labels/{name}", file_name + ".txt"), "w") as f:
        f.write(yolo_format)
        

def process_image_val(image_path):
    name = "val"
    os.makedirs(f"data/yolo/images/{name}",exist_ok=True)
    os.makedirs(f"data/yolo/labels/{name}",exist_ok=True)
    img, yolo_format, file_name = create_img(image_path)
    img.save(os.path.join(f"data/yolo/images/{name}", file_name + ".jpg"))
    
    with open(os.path.join(f"data/yolo/labels/{name}", file_name + ".txt"), "w") as f:
        f.write(yolo_format)

with Pool(CORES) as pool:
        list(tqdm(pool.imap(process_image_train, train_imgs * TIMES), total=len(train_imgs * TIMES)))
        

with Pool(CORES) as pool:
        list(tqdm(pool.imap(process_image_val, val_imgs * TIMES), total=len(val_imgs * TIMES)))
        




def zip_directory(folder_path, output_filename):
    """
    Creates a zip archive from a directory.
    
    Parameters:
    - folder_path: Path to the directory to be zipped.
    - output_filename: The output filename for the zip archive without the extension.
    
    The zip archive will be created in the same location as the output_filename.
    """
    # The root_dir parameter is where to look for the folder to be zipped.
    # The base_dir parameter is the directory to be zipped.
    # If your folder_path is "/path/to/folder", root_dir would be "/path/to" and base_dir would be "folder"
    root_dir, base_dir = os.path.split(folder_path)
    
    # Create the zip archive (without the .zip extension, as it's added automatically)
    shutil.make_archive(output_filename, 'zip', root_dir, base_dir)
    print(f"Directory '{folder_path}' has been zipped into '{output_filename}.zip'")

zip_directory("data/yolo", "data/yolo")