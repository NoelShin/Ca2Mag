import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

dataset_dir = './datasets/Over_0_std'
label_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Input', '*.png')))
target_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Target', '*.png')))
list_label = list()

for path in tqdm(label_path_list):
    list_label.append(np.array(Image.open(path)) / 255.0)

print("Mean: ", np.mean(list_label), "Std: ", np.std(list_label))