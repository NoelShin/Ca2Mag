if __name__ == '__main__':
    import os
    import sys
    from glob import glob
    from PIL import Image
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    STD = 0
    IMAGE_SIZE = 1024
    ITERATION = 470000
    MODEL_NAME = 'height_1024_val_False_without_error'

    dir_test_image = './checkpoints/Over_{}_std/Image/Test/{}/{}'.format(str(STD), MODEL_NAME, str(ITERATION))
    dir_model = './checkpoints/Over_{}_std/Model/{}'.format(str(STD), MODEL_NAME)

    list_fake = sorted(glob(os.path.join(dir_test_image, '*_fake.png')))
    list_real = sorted(glob(os.path.join(dir_test_image, '*_real.png')))

    list_diff = list()
    list_abs_diff = list()
    dict_sum_abs_diff = {'name': 'default', 'value': sys.maxsize}
    assert len(list_fake) == len(list_real)
    for i in tqdm(range(len(list_fake))):
        name = os.path.splitext(os.path.split(list_fake[i])[-1])[0].strip('_fake')
        name = name + '_diff.png'
        fake = ((np.array(Image.open(list_fake[i])) / 255) - 0.5) * 200
        real = ((np.array(Image.open(list_real[i])) / 255) - 0.5) * 200
        diff = real - fake


        naive_image = diff + 200.0
        naive_image /= 400.0
        naive_image *= 255.0
        naive_image = naive_image.astype(np.uint8)

        Image.fromarray(naive_image).save(os.path.join(dir_test_image, name))

        for j in range(IMAGE_SIZE):
            for k in range(IMAGE_SIZE):
                if (j - 511) ** 2 + (k - 511) ** 2 > 392 ** 2:
                    diff[j, k] = 0.
                else:
                    list_diff.append(diff[j, k])
                    list_abs_diff.append(abs(diff[j, k]))

        if np.sum(list_abs_diff) < dict_sum_abs_diff['value']:
            dict_sum_abs_diff.update({'name': name, 'value': np.sum(list_abs_diff)})

        # font = {'weight': 'bold', 'size': 14}
        # matplotlib.rc('font', **font)
        #
        # plt.figure(figsize=[12, 12])
        # plt.margins(0.)
        # plt.xticks(ticks=[0, 512, 1024])
        # plt.yticks(ticks=[0, 512, 1024])
        # plt.xlabel('X axis', fontdict={'weight': 'bold', 'size': 14})
        # plt.ylabel('Y axis', fontdict={'weight': 'bold', 'size': 14})
        # plt.title('Difference map between generated and real HMI', fontdict={'weight': 'bold', 'size': 18})
        # plt.imshow(diff, cmap='gray')
        # plt.colorbar(ticks=[-200, -100, 0, 100, 200])
        # plt.savefig(os.path.join(dir_test_image, name))
        # plt.close()

    print("Name and value of Minimum difference: {} {}".format(*dict_sum_abs_diff.values()))
    matplotlib.rc('font', weight='bold', size=14)
    plt.figure(figsize=[12, 8])
    plt.xlabel('Total unsigned magnetic field difference', fontdict={'weight': 'bold', 'size': 14})
    plt.ylabel('Portion of given difference', fontdict={'weight': 'bold', 'size': 14})
    print("mean: {}, std: {}".format(np.mean(dict_sum_abs_diff.values()), np.std(dict_sum_abs_diff.values())))
    # plt.text(150, 0.035, r'$\mu={},\ \sigma={}$'.format())
    plt.grid(True)
    n, bins, patches = plt.hist(list_diff, bins=range(-200, 205, 5), density=True)
    plt.savefig(os.path.join(dir_model, '{}_difference_map_histogram.png'.format(ITERATION)))

    with open(os.path.join(dir_model, '{}_difference_map_histogram.txt'.format(ITERATION)), 'wt') as log:
        for i in sorted(list_diff):
            log.write(str(i) + '\n')
        log.close()


