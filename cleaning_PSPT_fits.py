if __name__ == '__main__':
    import os
    from glob import glob

    dir_fits = './datasets/Fits/PSPT/2015'
    list_gz = glob(os.path.join(dir_fits, '*.gz'))
    print(len(list_gz))
    for gz in list_gz:
        os.remove(gz)