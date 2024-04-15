import pandas as pd
import random 
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageOps
from multiprocessing import Pool
import random
random.seed(0)
np.random.seed(0)


N_TOTAL = 100000
CORES = 8

df = pd.read_csv("data/parsed_from_index.csv", index_col=0).drop_duplicates("name")
df.primary.value_counts()

def sample_even(n_times):
    
    samples = []
    to_class = df.groupby("primary")["name"].apply(lambda x: list(x))
    with_name = df.set_index("name")
    
    for _ in tqdm(range(n_times)):

        index = random.choice(random.choice(to_class.values))
        row = with_name.loc[index]
        samples.append(row)
    return pd.DataFrame(samples)

samples = sample_even(N_TOTAL)
samples.value_counts("primary", normalize=True).plot.bar()
plt.show()
samples["y_center"] = np.random.uniform(0.15, 0.85, len(samples))
samples["x_center"] = np.random.uniform(0.15, 0.85, len(samples))

samples["width"] = np.random.uniform(0.1, 0.30, len(samples))


background_image = glob("data/places_imgs/**/*.jpg", recursive=True)
samples["background_path"] = np.random.choice(background_image, len(samples))

samples["Id"] = np.arange(len(samples))
# samples.to_csv("data/test_generated_imgs.csv")
samples = pd.read_csv("data/test_generated_imgs.csv", index_col=0)



def turn_tranparent(img):
    img = img.convert("RGBA")

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img

os.makedirs("data/generated", exist_ok=True)
os.makedirs("data/generated_labels", exist_ok=True)

def generate_img(row : pd.Series):
    labels_path = f"data/generated_labels/{row.Id}.txt"
    save_path = f"data/generated/{row.Id}.jpg"
    
    if os.path.exists(save_path):
        return
    
    
    img = Image.open(row.background_path)
    og_width, _ = img.size
    new_width = min(og_width, 400) # make to 4/3 format
    new_height = int(new_width / 4 * 3)
    
    img = img.resize((new_width, new_height)).convert("RGBA")
    back_ground = Image.new("RGBA", img.size)

    pokemon = Image.open(row.image_path)
    
    if np.random.randint(0,2) > 0:
        pokemon = ImageOps.mirror(pokemon)
    pokemon = turn_tranparent(pokemon)

    resize_x = int(1 / (pokemon.size[0] /  new_width  / row.width) * pokemon.size[0])
    resize_y = int(1 / (pokemon.size[1] / new_height / row.width) * pokemon.size[1])

    pokemon = pokemon.resize((resize_x, resize_y))
    
    x_width = resize_x/new_width
    y_width = resize_y/new_height

    upper_corner = int((row.x_center - x_width/2)*new_width)
    left_corner = int((row.y_center - y_width/2)*new_height)
    back_ground.paste(pokemon, (upper_corner, left_corner), pokemon.split()[3])
    img = Image.alpha_composite(img, back_ground).convert("RGB")
    img.save(save_path)
    
    with open(labels_path, "w") as f:
        f.write(" ".join(list(map(str, [0, row.x_center, row.y_center, x_width, y_width]))))

def wrapper(rows):
    try:
        generate_img(rows)
    except Exception as e:
        print(e)

rows = [samples.iloc[i] for i in range(len(samples))]

with Pool(CORES) as pool:
        list(tqdm(pool.imap(wrapper, rows), total=len(rows)))
        



