if __name__ == '__main__':
    import os
    import sys
    from glob import glob
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from utils import binning_and_cal_pixel_cc


    def uint8_to_gauss(np_array):
        np_array = np_array.astype(np.float64)
        np_array -= 127.5
        np_array /= 1.275
        assert all(np_array.flatten() >= -100.0) and all(np_array.flatten() <= 100.0), print(np_array.min(), np_array.max())

        return np_array


    THRESHOLD = 0
    MODEL = 'height_1024_val_False_without_error'
    ITERATION = 470000
    PATCH_SIZE = 128

    dir_analysis = './checkpoints/Over_{}_std/Analysis/{}/{}/Patch'.format(THRESHOLD, MODEL, ITERATION)
    dir_patch_active = './checkpoints/Over_{}_std/Analysis/{}/{}/Patch/Active'.format(THRESHOLD, MODEL, ITERATION)
    dir_patch_quiet = './checkpoints/Over_{}_std/Analysis/{}/{}/Patch/Quiet'.format(THRESHOLD, MODEL, ITERATION)
    list_path_active_real = sorted(glob(os.path.join(dir_patch_active, '*_real.png')))
    list_path_active_fake = sorted(glob(os.path.join(dir_patch_active, '*_fake.png')))

    list_path_quiet_real = sorted(glob(os.path.join(dir_patch_quiet, '*_real.png')))
    list_path_quiet_fake = sorted(glob(os.path.join(dir_patch_quiet, '*_fake.png')))

    list_TUMF_fake = list()
    list_TUMF_real = list()

    list_cc_1x1_fake = list()
    list_cc_1x1_real = list()
    list_cc_1x1 = list()
    list_cc_bin_2x2 = list()
    list_cc_bin_4x4 = list()
    list_cc_bin_8x8 = list()
    list_R1 = list()
    list_R2 = list()
    dict_min_cc = {'name': None, 'val': 1.0}
    dict_max_cc = {'name': None, 'val': 0.0}
    for i in tqdm(range(len(list_path_active_fake))):
        real = uint8_to_gauss(np.array(Image.open(list_path_active_real[i])))
        fake = uint8_to_gauss(np.array(Image.open(list_path_active_fake[i])))

        carrier_fake = list()
        carrier_real = list()
        for j in range(PATCH_SIZE):
            for k in range(PATCH_SIZE):
                if abs(fake[j, k]) >= 10:
                    carrier_fake.append(abs(fake[j, k]))
                if abs(real[j, k]) >= 10:
                    carrier_real.append(abs(real[j, k]))

        TUMF_fake, TUMF_real = np.array(carrier_fake).sum(), np.array(carrier_real).sum()
        list_TUMF_fake.append(TUMF_fake)
        list_TUMF_real.append(TUMF_real)

        list_R1.append((TUMF_fake - TUMF_real) / TUMF_real)
        list_R2.append(((fake.flatten() - real.flatten()) ** 2).sum() / (real.flatten() ** 2).sum())
        # print(((fake.flatten() - real.flatten()) ** 2).sum() / (real.flatten() ** 2).sum())

        list_cc_1x1.append(np.corrcoef(fake.flatten(), real.flatten())[0][1])
        list_cc_bin_2x2.append(binning_and_cal_pixel_cc(fake, real, 2, patch=True))
        list_cc_bin_4x4.append(binning_and_cal_pixel_cc(fake, real, 4, patch=True))
        list_cc_bin_8x8.append(binning_and_cal_pixel_cc(fake, real, 8, patch=True))
        if list_cc_bin_8x8[-1] < dict_min_cc["val"]:
            dict_min_cc.update({"name": os.path.split(list_path_active_fake[i])[-1], 'val': list_cc_bin_8x8[-1]})
        if list_cc_bin_8x8[-1] > dict_max_cc["val"]:
            dict_max_cc.update({"name": os.path.split(list_path_active_fake[i])[-1], 'val': list_cc_bin_8x8[-1]})
    print(dict_min_cc["name"], dict_min_cc["val"])
    print(dict_max_cc["name"], dict_max_cc["val"])


    list_cc_bin_8x8.sort()
    print("median valud: ", list_cc_bin_8x8[1092])
    cc_TUMF = np.corrcoef(np.array(list_TUMF_fake), np.array(list_TUMF_real))[0][1]
    cc_1x1 = np.mean(list_cc_1x1)
    cc_bin_2x2 = np.mean(list_cc_bin_2x2)
    cc_bin_4x4 = np.mean(list_cc_bin_4x4)
    cc_bin_8x8 = np.mean(list_cc_bin_8x8)

    plt.figure()
    plt.rc("axes", axisbelow=True)
    plt.hist(list_cc_bin_8x8, bins=list(np.arange(-1., 1.05, 0.05)))
    plt.title("Active region 8x8 binning CC histogram")
    plt.ylabel("# of samples")
    plt.xlabel("Correlation Coefficient")
    plt.grid(True)
    plt.text(-0.95, 230, "mean    = 0.55\nmedian = 0.65",
             size='medium',
             weight='bold',
             bbox=dict(facecolor='none', boxstyle='round'))
    plt.show()

    R1_mean = np.mean(list_R1)
    R1_std = np.std(list_R1)
    R2_mean = np.mean(list_R2)
    R2_std = np.std(list_R2)

    with open(os.path.join(dir_analysis, 'measurements_active_regions.txt'), 'wt') as analysis:
        analysis.write('CorrCoef_TUMF, CorrCoef_1x1, CorrCoef_2x2, CorrCoef_4x4, CorrCoef_8x8, '
                       'R1_mean, R1_std, R2_mean, R2_std\n')
        analysis.write(str(cc_TUMF) + ', ' + str(cc_1x1) + ', ' +
                       str(cc_bin_2x2) + ', ' + str(cc_bin_4x4) + ', ' + str(cc_bin_8x8) + ', ' +
                       str(R1_mean) + ', ' + str(R1_std) + ', ' + str(R2_mean) + ', ' + str(R2_std))
        analysis.close()

    # Quiet region
    list_TUMF_fake = list()
    list_TUMF_real = list()

    list_cc_1x1_fake = list()
    list_cc_1x1_real = list()
    list_cc_1x1 = list()
    list_cc_bin_2x2 = list()
    list_cc_bin_4x4 = list()
    list_cc_bin_8x8 = list()
    list_R1 = list()
    list_R2 = list()

    for i in tqdm(range(len(list_path_quiet_fake))):
        real = np.array(Image.open(list_path_quiet_real[i]), dtype=np.float64)
        fake = np.array(Image.open(list_path_quiet_fake[i]), dtype=np.float64)

        real -= 127.5
        real /= 127.5
        real *= 100.0
        real = real.clip(-100., 100.)

        fake -= 127.5
        fake /= 127.5
        fake *= 100.0
        fake = fake.clip(-100., 100.)

        carrier_fake = list()
        carrier_real = list()
        for j in range(PATCH_SIZE):
            for k in range(PATCH_SIZE):
                if abs(fake[j, k]) >= 10:
                    carrier_fake.append(abs(fake[j, k]))
                if abs(real[j, k]) >= 10:
                    carrier_real.append(abs(real[j, k]))

        TUMF_fake, TUMF_real = np.array(carrier_fake).sum(), np.array(carrier_real).sum()
        list_TUMF_fake.append(TUMF_fake)
        list_TUMF_real.append(TUMF_real)

        list_R1.append((TUMF_fake - TUMF_real) / TUMF_real)
        list_R2.append(((fake.flatten() - real.flatten()) ** 2).sum() / (real.flatten() ** 2).sum())

        list_cc_1x1.append(np.corrcoef(fake.flatten(), real.flatten())[0][1])
        list_cc_bin_2x2.append(binning_and_cal_pixel_cc(fake, real, 2, patch=True))
        list_cc_bin_4x4.append(binning_and_cal_pixel_cc(fake, real, 4, patch=True))
        list_cc_bin_8x8.append(binning_and_cal_pixel_cc(fake, real, 8, patch=True))

    cc_TUMF = np.corrcoef(np.array(list_TUMF_fake), np.array(list_TUMF_real))[0][1]
    cc_1x1 = np.mean(list_cc_1x1)
    cc_bin_2x2 = np.mean(list_cc_bin_2x2)
    cc_bin_4x4 = np.mean(list_cc_bin_4x4)
    cc_bin_8x8 = np.mean(list_cc_bin_8x8)

    R1_mean = np.mean(list_R1)
    R1_std = np.std(list_R1)
    R2_mean = np.mean(list_R2)
    R2_std = np.std(list_R2)

    with open(os.path.join(dir_analysis, 'measurements_quiet_regions.txt'), 'wt') as analysis:
        analysis.write('CorrCoef_TUMF, CorrCoef_1x1, CorrCoef_2x2, CorrCoef_4x4, CorrCoef_8x8, '
                       'R1_mean, R1_std, R2_mean, R2_std\n')
        analysis.write(str(cc_TUMF) + ', ' + str(cc_1x1) + ', ' +
                       str(cc_bin_2x2) + ', ' + str(cc_bin_4x4) + ', ' + str(cc_bin_8x8) + ', ' +
                       str(R1_mean) + ', ' + str(R1_std) + ', ' + str(R2_mean) + ', ' + str(R2_std))
        analysis.close()