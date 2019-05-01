if __name__ == '__main__':
    import os
    from glob import glob
    from shutil import copy
    import numpy as np

    image_root = './Images/PSPT'
    dir_dst = './Images/PSPT/DailyBest'
    os.makedirs(dir_dst, exist_ok=True)
    list_raw_paths = list()
    list_grad_paths = list()
    list_HMI = sorted(glob(os.path.join('./Images/HMI_100', '*.png')))
    list_dir = sorted(os.listdir(image_root))

    for dir in list_dir:
        list_raw_paths.extend(sorted((glob(os.path.join(image_root, dir, 'Raw', '*.png')))))
        list_grad_paths.extend(sorted((glob(os.path.join(image_root, dir, 'Gradient/2', '*.png')))))

    list_dates = list()
    for i in range(len(list_raw_paths)):
        name = os.path.split(os.path.splitext(list_raw_paths[i])[0])[-1]
        list_dates.append(name[:8])

    tuple_dates = sorted(frozenset(list_dates))
    for date in tuple_dates:
        list_raw_same_date = list()
        list_grads_same_date = list()
        switch = False
        for i, raw in enumerate(list_raw_paths):
            if raw.find(date) != -1:
                list_raw_same_date.append(list_raw_paths[i])
                list_grads_same_date.append(np.fromfile(list_grad_paths[i]).sum())
                switch = True
            else:
                if not switch:
                    continue
                else:
                    break

        np_grads_same_date = np.asarray(list_grads_same_date)
        index = np_grads_same_date.argmax()
        print(os.path.splitext(os.path.split(list_raw_same_date[index])[-1])[0][:15])

        copy(list_raw_same_date[index], dir_dst)
