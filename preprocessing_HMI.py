if __name__ == '__main__':
    import os
    from astropy.io import fits
    from PIL import Image
    from glob import glob
    import numpy as np
    from time import time
    from tqdm import tqdm

    st = time()

    year = '2013'
    months = ['09']  # ['01', '02', '03', '04', '05', '06']
    resample = Image.NEAREST
    dir_dst = '/DATA/RAID/Noel/Datasets/HMI/Image/{}'.format(year)
    os.makedirs(dir_dst) if not os.path.isdir(dir_dst) else None

    list_HMI = list()
    for month in months:
        dir_HMI = '/DATA/RAID/Noel/Datasets/HMI/{}/{}'.format(year, month)
        days = os.listdir(dir_HMI)
        for day in ['06', '09', '10', '11', '13', '16', '17', '22', '24', '27', '29']:  # days:
            list_HMI.extend(sorted(glob(os.path.join(dir_HMI, day, '*.fits'))))
    for HMI in tqdm(list_HMI):
        name = os.path.splitext(os.path.split(HMI)[-1])[0]
        try:
            hdu = fits.open(HMI)
            hdu.verify('fix')  # I need to declare this for opening data.

            header = hdu[1].header  # Note that SDO's compressed data is accessed with index 1 not 0

            R_SUN_arc = header['RSUN_OBS']  # in arcsecond. Remember HMI has 0.5 arcsec per pixel
            R_SUN = R_SUN_arc / 0.5
            NAXIS = header['NAXIS1']
            CENTER_X = int(header['CRPIX1'])
            CENTER_Y = int(header['CRPIX2'])

            ratio = (392. + 5.0) / R_SUN  # 1024x1024  + 4.0 is to get rid of the limb noise.

            data = hdu[1].data
            data = data[CENTER_Y - 1 - int(R_SUN): CENTER_Y - 1 + int(R_SUN),
                   CENTER_X - 1 - int(R_SUN): CENTER_X - 1 + int(R_SUN)]

            if data.shape != (4096, 4096):
                pad = (4096 - data.shape[0]) // 2
                if (4096 - data.shape[0]) % 2 == 0:
                    data = np.pad(data, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
                else:
                    data = np.pad(data, ((pad + 1, pad), (pad + 1, pad)), 'constant', constant_values=0)
            assert data.shape == (4096, 4096)

            data = np.clip(data, -100, 100)
            data -= np.nanmin(data)
            data /= np.nanmax(data)
            data *= 255.0
            data = np.nan_to_num(data)
            data = data.astype(np.uint8)

            data = Image.fromarray(data)
            data = data.resize((int(NAXIS * ratio), int(NAXIS * ratio)))
            data = np.array(data)

            if data.shape != (1024, 1024):
                pad = (1024 - data.shape[0]) // 2
                if (1024 - data.shape[0]) % 2 == 0:
                    data = np.pad(data, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
                else:
                    data = np.pad(data, ((pad + 1, pad), (pad + 1, pad)), 'constant', constant_values=0)
            assert data.shape == (1024, 1024)
            data = np.fliplr(data)
            for i in range(1024):
                for j in range(1024):
                    if (i - 511) ** 2 + (j - 511) ** 2 > 392 ** 2:
                        data[i, j] = 0
            # data[511 - 392: 511 + 392, 511] = 255
            # data[511, 511 - 392: 511 + 392] = 255
            data = Image.fromarray(data)
            data.save(os.path.join(dir_dst, name + '.png'))

            del hdu, header, data

        except TypeError:
            with open('{}.txt'.format(year), 'wt' if not os.path.isfile('{}.txt'.format(year)) else 'a') as log:
                log.write('{}\n'.format(name))

        except OSError:
            with open('{}.txt'.format(year), 'wt' if not os.path.isfile('{}.txt'.format(year)) else 'a') as log:
                log.write('{}\n'.format(name))

    print(time() - st)