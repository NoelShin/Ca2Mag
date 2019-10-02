import os
from os.path import join
from astropy.io import fits
import numpy as np
from scipy.ndimage import rotate
from PIL import Image
from tqdm import tqdm
from skimage.morphology import disk
from skimage.filters.rank import gradient


# list_years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
list_years = ['2010', '2011', '2012', '2013', '2014', '2015']

for year in list_years:
    dir_fits = './datasets/Fits/PSPT/' + year
    dir_images = './datasets/Images/PSPT/' + year + '/Raw'
    dir_grads = './datasets/Images/PSPT/' + year + '/Gradient'
    list_grads = [2, 3, 4, 5]  # list of disk radii for calculating gradients. 5 is usually used.

    os.makedirs(dir_images, exist_ok=True)
    os.makedirs(dir_grads, exist_ok=True)

    for g in list_grads:
        os.makedirs(join(dir_grads, str(g))) if not os.path.isdir(join(dir_grads, str(g))) else None
        with open(join(dir_grads, str(g), 'grads_sum.txt'), 'wt') as log:
            log.write('Name, Gradient_sum\n')
            log.close()

    list_path_fits = sorted(os.listdir(dir_fits))

    for name in tqdm(list_path_fits):
        hdul = fits.open(join(dir_fits, name))
        header = hdul[0].header
        # Indexing starts at (1, 1) lower left corner

        EXPTIME = float(header['EXPTIME'])

        X0 = int(header['X0'])  # X offset into image
        Y0 = int(header['Y0'])  # Y offset into image

        NAXIS1 = int(header['NAXIS1'])
        NAXIS2 = int(header['NAXIS2'])

        XRADIUS = float(header['XRADIUS'])  # X radius
        YRADIUS = float(header['YRADIUS'])  # Y radius

        ratio = (392. / YRADIUS, 392. / XRADIUS)  # Ratio regarding HMI radius 392 (for 1024x1024 case)

        theta = float(header['SOLAR_P0'])

        data = hdul[0].data
        data /= EXPTIME
        # Centering image
        max_val = 0
        XRADIUS_int = int(np.ceil(XRADIUS))
        YRADIUS_int = int(np.ceil(YRADIUS))

        # Centering
        x = 0
        y = 0
        for i in range(NAXIS1 - XRADIUS_int * 2):
            box_sum = data[0: YRADIUS_int * 2, i: i + XRADIUS_int * 2].sum()
            if box_sum > max_val:
                max_val = box_sum
                x = i
        max_val = 0
        for i in range(NAXIS2 - YRADIUS_int * 2):
            box_sum = data[i: i + YRADIUS_int * 2, x: x + XRADIUS_int * 2].sum()
            if box_sum > max_val:
                max_val = box_sum
                y = i

        data = data[y: y + 2 * YRADIUS_int, x: x + 2 * XRADIUS_int]

        # Correcting polar coordinates up to top of the image.
        data = rotate(data, theta, reshape=False)
        image = Image.fromarray(data)

        # Resizing the shape so that the radius has 392 value.
        shape = (int(np.around(data.shape[1] * ratio[1])), int(np.around(data.shape[0] * ratio[0])))
        image = image.resize(shape)

        np_image = np.array(image)
        pad_height = (1024 - np_image.shape[0]) // 2
        pad_width = (1024 - np_image.shape[1]) // 2
        np_image = np.pad(np_image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

        if not np_image.shape[0] == 1024:
            padding_height = 1024 - np_image.shape[0]
            if padding_height % 2 == 0:
                np_image = np.pad(np_image, ((padding_height // 2, padding_height // 2), (0, 0)), mode='constant')
            else:
                assert padding_height == 1
                np_image = np.pad(np_image, ((padding_height, 0), (0, 0)), mode='constant')

        if not np_image.shape[1] == 1024:
            padding_width = 1024 - np_image.shape[1]
            if padding_width % 2 == 0:
                np_image = np.pad(np_image, ((0, 0), (padding_width // 2, padding_width // 2)), mode='constant')
            else:
                assert padding_width == 1
                np_image = np.pad(np_image, ((0, 0), (padding_width, 0)), mode='constant')

        assert np_image.shape == (1024, 1024), print(name, np_image.shape)
        np_image -= np_image.min()
        np_image = np_image / np_image.max()
        np_image *= 255.0
        np_image = np_image.astype(np.uint8)
        np_image = np.flipud(np_image)
        image = Image.fromarray(np_image)
        image.save(join(dir_images, name[:13].replace('.', '_') + '.png'))

        for g in list_grads:
            grad = gradient(np_image, disk(g))
            Image.fromarray(grad).save(join(dir_grads, str(g), name[:13].replace('.', '_') + '_grad_{}.png'.format(g)))
            with open(join(dir_grads, str(g), 'grads_sum.txt'), 'a') as log:
                log.write(name[:13].replace('.', '_') + ', ' + str(grad.sum()) + '\n')
                log.close()
