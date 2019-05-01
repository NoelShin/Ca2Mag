if __name__ == '__main__':
    import os
    from astropy.io import fits
    from PIL import Image
    from glob import glob
    import numpy as np
    from time import time

    st = time()

    year = '2011'
    months = ['01', '02', '03', '04', '05', '06']
    resample = Image.NEAREST
    dir_dst = '/DATA/RAID/Noel/Datasets/HMI/Image/{}'.format(year)

    list_HMI = list()
    for month in months:
        dir_HMI = '/DATA/RAID/Noel/Datasets/HMI/{}/{}'.format(year, month)
        days = os.listdir(dir_HMI)
        for day in days:
            list_HMI.extend(sorted(glob(os.path.join(dir_HMI, day, '*.fits'))))
    print(len(list_HMI))
    for HMI in list_HMI:
        name = os.path.splitext(os.path.split(HMI)[-1])[0]
        hdu = fits.open(HMI)
        hdu.verify('fix')  # I need to declare this for opening data.

        header = hdu[1].header  # Note that SDO's compressed data is accessed with index 1 not 0
        R_SUN = header['RSUN_OBS']
        NAXIS = header['NAXIS1']

        ratio = 196./R_SUN  # 1024x1024

        data = hdu[1].data
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

        data = Image.fromarray(data)
        data.save(os.path.join(dir_dst, name + '.png'))

    print(time() - st)