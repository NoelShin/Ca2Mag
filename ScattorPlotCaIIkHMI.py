import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm

DIR_CAIIK = "/Users/noel/Desktop/*.png"
DIR_HMI = "/Users/noel/Desktop/*.npy"
DST_SAVE = "/Users/noel/Desktop"

IMAGE_SIZE = 1024
R_SUN = 392

k = 0
listCircleIndex = list()
for i in range(IMAGE_SIZE):
    for j in range(IMAGE_SIZE):
        if (i - (IMAGE_SIZE // 2) - 1) ** 2 + (j - (IMAGE_SIZE // 2) - 1) ** 2 <= R_SUN ** 2:
            listCircleIndex.append(k)
        k += 1


listCallk = sorted(glob(DIR_CAIIK))
listHMI = sorted(glob(DIR_HMI))

assert len(listCallk) == len(listHMI)

dictCallkHMI = dict()
for i in tqdm(range(len(listHMI))):
    callk = np.array(Image.open(listCallk[i]), dtype=np.uint8).flatten()
    hmi = np.load(listHMI[i]).astype(dtype=np.float64).flatten()
    hmi = np.abs(hmi)  # For making it unsigned.

    for j in tqdm(listCircleIndex):
        k, v = callk[j], hmi[j]
        if k in dictCallkHMI.keys():
            cnt = dictCallkHMI[k][0]
            dictCallkHMI[k].append(v)
            dictCallkHMI[k][0] = cnt + 1
        else:
            dictCallkHMI.update({k: [0, v]})

listValCallk, listValHMI = list(), list()
for k, v in dictCallkHMI.items():
    listValCallk.append(k)
    listValHMI.append(np.mean(v[1:]))  # Exclude the 0-th index in v as it means # of the same Ca ll K vals.


fig, ax = plt.subplots(1)
plt.scatter(listValCallk, listValHMI, color='b', s=4)

fontdict = {"weight": "bold"}
plt.xlabel("Ca II K (pixel intensity)", fontdict=fontdict)
plt.xticks([0, 50, 100, 150, 200, 250])
ax.xaxis.set_minor_locator(AutoMinorLocator())

plt.ylabel("Unsigned magnetic flux (Gauss)", fontdict=fontdict)
plt.yscale("log")
plt.yticks([1, 10, 100, 1000])

ax.tick_params(axis="both", which="both", direction="in")
plt.show()

np.save("{}/dictCallKHMI.npy".format(DST_SAVE), dictCallkHMI)