import os
from os.path import join
from astropy.io import fits
import numpy as np
from scipy.ndimage import rotate
from PIL import Image
from tqdm import tqdm

dir_fits = './datasets/Fits/VSM'
dir_images = './datasets/Images/VSM'
os.makedirs(dir_images) if not os.path.isdir(dir_images) else None

list_fits_name = sorted(os.listdir(dir_fits))[1789:]
list_error_fits = list()
for name in tqdm(list_fits_name):
    path = join(dir_fits, name)
    hdul = fits.open(path)
    try:
        header = hdul[0].header
        data = hdul[0].data
        # print(repr(header))
    except TypeError:
        print(name)
        list_error_fits.append(name)
        continue

    NAXIS1 = int(header['NAXIS1'])
    NAXIS2 = int(header['NAXIS2'])

    CENTER_X = int(np.around(float(header['IMG_X0'])))
    CENTER_Y = int(np.around(float(header['IMG_Y0'])))

    RADIUS = float(header['IMG_R0'])
    RADIUS_int = int(np.ceil(float(header['IMG_R0'])))
    ratio = 392. / RADIUS

    # P0 = float(header['OBS_P'])

    # Centering
    data = data[CENTER_Y - RADIUS_int - 20: CENTER_Y + RADIUS_int + 21, CENTER_X - RADIUS_int - 20:
                                                                        CENTER_X + RADIUS_int + 21]
    original_width, original_height = data.shape[1], data.shape[0]

    # Resizing
    data = Image.fromarray(data).resize((int(np.around(original_width * ratio)),
                                         int(np.around(original_height * ratio))))
    data = np.array(data)

    # Rotating
    # data = rotate(data, -P0, reshape=False)

    # Padding
    width, height = data.shape[1], data.shape[0]
    pad_width = (1024 - width) // 2
    pad_height = (1024 - height) // 2
    data = np.pad(data, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
    width, height = data.shape
    if width != 1024:
        pad = 1024 - width
        if pad % 2 == 0:
            data = np.pad(data, ((0, 0), (pad // 2, pad // 2)), 'constant')
        else:
            assert pad == 1, print(name)
            data = np.pad(data, ((0, 0), (pad, 0)), 'constant')

    if height != 1024:
        pad = 1024 - height
        if pad % 2 == 0:
            data = np.pad(data, ((pad // 2, pad // 2), (0, 0)), 'constant')
        else:
            assert pad == 1, print(name)
            data = np.pad(data, ((pad, 0), (0, 0)), 'constant')

    assert data.shape == (1024, 1024)

    data -= data.min()
    data = data / data.max()
    data *= 255.0
    data = data.astype(np.uint8)

    image = Image.fromarray(data)
    name = name.strip('k4v82_cont.fts')
    name = name.replace('t', '_')
    name = '20' + name + '.png'

    image.save(join(dir_images, name))
    del header, data, hdul, image

with open(join(dir_fits, 'ErrorFits.txt'), 'wt') as log:
    for name in list_error_fits:
        log.write(name + '\n')
    log.close()
