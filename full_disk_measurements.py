import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import binning_and_cal_pixel_cc
import matplotlib.pyplot as plt


def uint8_to_gauss(np_array):
    np_array = np_array.astype(np.float64)
    np_array -= 127.5
    np_array /= 1.275
    assert all(np_array.flatten() >= -100.0) and all(np_array.flatten() <= 100.0)
    return np_array

STD = 0
MODEL = 'pix2pix'
ITERATION = 470000

dir_image = './checkpoints/Over_{}_std/Image/Test/{}/{}'.format(STD, MODEL, ITERATION)

list_real = sorted(glob(os.path.join(dir_image, '*_real.png')))
list_fake = sorted(glob(os.path.join(dir_image, '*_fake.png')))

list_index = list()
k = 0
for i in range(1024):
    for j in range(1024):
        if (511 - i) ** 2 + (511 - j) ** 2 <= 392 ** 2:
            list_index.append(k)
        k += 1

list_TUMF_real, list_TUMF_fake = list(), list()
list_pixel_cc, list_8x8_cc = list(), list()
list_R1, list_R2 = list(), list()

for i in tqdm(range(len(list_real))):
    real = uint8_to_gauss(np.array(Image.open(list_real[i])))  # [list_index]
    fake = uint8_to_gauss(np.array(Image.open(list_fake[i])))  # [list_index]

    real_flatten, fake_flatten = real.flatten(), fake.flatten()
    real_disk, fake_disk = real_flatten[list_index], fake_flatten[list_index]

    TUMF_real = sum(abs(p) for p in real_disk if abs(p) >= 10.)
    TUMF_fake = sum(abs(p) for p in fake_disk if abs(p) >= 10.)

    list_TUMF_real.append(TUMF_real)
    list_TUMF_fake.append(TUMF_fake)

    list_R1.append((TUMF_fake - TUMF_real) / TUMF_real)
    list_R2.append(np.sum((fake_disk - real_disk) ** 2) / np.sum(real_disk ** 2))

    list_pixel_cc.append(np.corrcoef(real_disk, fake_disk)[0, 1])
    cc_8x8 = binning_and_cal_pixel_cc(fake, real, bin_size=8)
    list_8x8_cc.append(cc_8x8)

plt.hist(list_8x8_cc, bins=list(np.arange(-2.0, 2.0, 0.05)))
plt.show()
print("1x1 pixel corrcoef on the disk (mean, std): ", np.mean(list_pixel_cc), np.std(list_pixel_cc))
print("8x8 pixel corrcoef on the disk (mean, std): ", np.mean(list_8x8_cc), np.std(list_8x8_cc))
print("TUMF corrcoef on the disk: ", np.corrcoef(list_TUMF_real, list_TUMF_fake)[0, 1])
print("R1 error (mean, std) on the disk: ", np.mean(list_R1), np.std(list_R1))
print("R2 error (mean, std) on the disk: ", np.mean(list_R2), np.std(list_R2))