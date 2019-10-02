import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def my_click(event):
    positions.append((int(event.xdata), int(event.ydata)))


def my_close(event):
    event.close()

ACTIVE = True

STD = 0
ITERATION = 470000
MODEL_NAME = 'height_1024_val_False_without_error'

dir_image = './checkpoints/Over_{}_std/Image/Test/{}/{}'.format(STD, MODEL_NAME, ITERATION)
dir_patch_active = './checkpoints/Over_{}_std/Analysis/{}/{}/Patch/Active'.format(STD, MODEL_NAME, ITERATION)
dir_patch_quiet = './checkpoints/Over_{}_std/Analysis/{}/{}/Patch/Quiet'.format(STD, MODEL_NAME, ITERATION)
os.makedirs(dir_patch_active, exist_ok=True)
os.makedirs(dir_patch_quiet, exist_ok=True)

list_fake = sorted(glob(os.path.join(dir_image, '*_fake.png')))
list_real = sorted(glob(os.path.join(dir_image, '*_real.png')))

assert len(list_fake) == len(list_real)

# with open(os.path.join(dir_patch_active, 'positions.txt'), 'wt') as log:
#     log.write('file_name, positions(W, H)\n')
#     log.close()

dir_patch = dir_patch_active if ACTIVE else dir_patch_quiet

with open(os.path.join(dir_patch, 'positions.txt'), 'wt') as log:
    log.write('file_name, positions(W, H)\n')
    log.close()

list_index = list()
k = 0
for i in range(1024):
    for j in range(1024):
        if (i - 512) ** 2 + (j - 512) ** 2 > 392 ** 2:
            list_index.append(k)
        k += 1

for i in tqdm(range(len(list_fake))):
    name = os.path.split(list_fake[i])[-1].strip('_fake.png')
    fake_np = np.array(Image.open(list_fake[i]), dtype=np.uint8).flatten()
    real_np = np.array(Image.open(list_real[i]), dtype=np.uint8).flatten()

    fake_np[list_index] = 127
    real_np[list_index] = 127

    fake_image = fake_np.reshape((1024, 1024))
    real_image = real_np.reshape((1024, 1024))

    f, a = plt.subplots(figsize=(15, 15))
    a.imshow(real_image, cmap='gray')
    f.tight_layout()

    positions = list()

    cid = f.canvas.mpl_connect('button_press_event', my_click)
    plt.show()
    f.canvas.mpl_connect(cid, my_close)

    for j, position in enumerate(positions):
        fake_patch = Image.fromarray(np.array(fake_image)[position[1] - 63: position[1] + 65, position[0] - 63: position[0] + 65])
        real_patch = Image.fromarray(np.array(real_image)[position[1] - 63: position[1] + 65, position[0] - 63: position[0] + 65])

        fake_patch.save(os.path.join(dir_patch, name + '_patch_{}_fake.png'.format(j)))
        real_patch.save(os.path.join(dir_patch, name + '_patch_{}_real.png'.format(j)))

    with open(os.path.join(dir_patch, 'positions.txt'), 'a') as log:
        log.write(', '.join([name, *list(map(lambda x: str(x), positions))]) + '\n')
        log.close()

# #f.canvas.mpl_connect(cid, my_close)
