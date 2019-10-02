import os
from os.path import split, splitext
from glob import glob
from shutil import copy
from tqdm import tqdm

FACTOR_STD = 0
MONTH_PARTITION = ['01', '07']

dir_HMI = './Images/HMI_100'
dir_train_input = './Over_{}_std_01_07/Train/Input'.format(FACTOR_STD)
dir_train_target = './Over_{}_std_01_07/Train/Target'.format(FACTOR_STD)
dir_test_input = './Over_{}_std_01_07/Test/Input'.format(FACTOR_STD)
dir_test_target = './Over_{}_std_01_07/Test/Target'.format(FACTOR_STD)

os.makedirs(dir_train_input, exist_ok=True)
os.makedirs(dir_train_target, exist_ok=True)
os.makedirs(dir_test_input, exist_ok=True)
os.makedirs(dir_test_target, exist_ok=True)

list_PSPT = sorted(glob('./Images/PSPT/Aligned_{}_std/*.png'.format(FACTOR_STD)))
print("Total_PSPT(Train and Test): ", len(list_PSPT))
# list_HMI = list()
# for dir in sorted(os.listdir(dir_HMI)):
#     list_HMI.extend(sorted(glob(os.path.join(dir_HMI, dir, '*.png'))))
list_HMI = sorted(glob(os.path.join(dir_HMI, '*.png')))
print("Total HMI(training and test): ", len(list_HMI))

list_HMI_date = list()
list_HMI_time = list()
for HMI in list_HMI:
    list_HMI_date.append(split(splitext(HMI)[0])[-1][11:-2].replace('_', '')[:8])
    list_HMI_time.append(split(splitext(HMI)[0])[-1][11:-2].replace('_', ''))
    print(split(splitext(HMI)[0])[-1][3:].replace('_', ''))
    # print(HMI, split(splitext(HMI)[0])[-1][11:-2].replace('_', '')[:8])
    # print(HMI, split(splitext(HMI)[0])[-1][3:].replace('_', ''))

list_PSPT_date = list()
list_PSPT_time = list()
for PSPT in list_PSPT:
    list_PSPT_date.append(split(splitext(PSPT)[0])[-1].replace('_', '')[:8])
    list_PSPT_time.append(split(splitext(PSPT)[0])[-1].replace('_', ''))
    # print(split(splitext(PSPT)[0])[-1].replace('_', ''))

k = 0
for i, PSPT_date in enumerate(list_PSPT_date):
    if int(PSPT_date[:8]) >= 20110101:
        print("Index k: ", i)
        k = i
        break

# Get the closest time of HMI to PSPT
list_aligned_PSPT = list()
list_aligned_HMI = list()

for PSPT_time in tqdm(list_PSPT_time[k:]):
    if (int(PSPT_time[-2:]) // 6) % 2 == 0:
        close_HMI_time = str(int(PSPT_time) - int(PSPT_time[-2:]) % 6)
        if close_HMI_time[-2:] == '60':
            close_HMI_time = str(int(close_HMI_time) + 100)
            if close_HMI_time[-4:-2] == '24':
                close_HMI_time = str(int(close_HMI_time) + 10000)

    else:
        close_HMI_time = str(int(PSPT_time) + (6 - int(PSPT_time[-2:]) % 6))
        if close_HMI_time[-2:] == '60':
            close_HMI_time = str(int(close_HMI_time) + 100)
            if close_HMI_time[-4:-2] == '24':
                close_HMI_time = str(int(close_HMI_time) + 10000)

    for HMI_time in list_HMI_time:
        if HMI_time == close_HMI_time:
            list_aligned_PSPT.append(PSPT_time)
            list_aligned_HMI.append(HMI_time)

print("Aligned_PSPT: ", len(list_aligned_PSPT), "Aligned_HMI: ", len(list_aligned_HMI))

list_PSPT_train = list()
list_PSPT_test = list()
list_HMI_train = list()
list_HMI_test = list()

for i in range(len(list_aligned_PSPT)):
    if list_aligned_PSPT[i][4:6] in MONTH_PARTITION:
        list_PSPT_test.append(list_aligned_PSPT[i])
        list_HMI_test.append(list_aligned_HMI[i])
    else:
        list_PSPT_train.append(list_aligned_PSPT[i])
        list_HMI_train.append(list_aligned_HMI[i])

for i in range(len(list_HMI_train) - 1):
    if list_HMI_train[i] == list_HMI_train[i + 1]:
        print(list_HMI_train[i])

print("# of train PSPT/HMI images: {}, {}, # of test images: {}, {}"
      .format(len(list_PSPT_train), len(frozenset(list_HMI_train)), len(list_PSPT_test), len(frozenset(list_HMI_test))))

for i in tqdm(range(len(list_PSPT_train))):
    for path in list_PSPT:
        switch = 0
        if split(splitext(path)[0])[-1].replace('_', '') == list_PSPT_train[i]:
            copy(path, dir_train_input)
            switch = 1
        else:
            if switch == 1:
                break
            else:
                continue

    for path in list_HMI:
        switch = 0
        if split(splitext(path)[0])[-1][11:-2].replace('_', '') == list_HMI_train[i]:
            copy(path, dir_train_target)
            switch = 1
        else:
            if switch == 1:
                break
            else:
                continue


for i in tqdm(range(len(list_PSPT_test))):
    for path in list_PSPT:
        switch = 0
        if split(splitext(path)[0])[-1].replace('_', '') == list_PSPT_test[i]:
            copy(path, dir_test_input)
            switch = 1
        else:
            if switch == 1:
                break
            else:
                continue

    for path in list_HMI:
        switch = 0
        if split(splitext(path)[0])[-1][11:-2].replace('_', '') == list_HMI_test[i]:
            copy(path, dir_test_target)
            switch = 1
        else:
            if switch == 1:
                break
            else:
                continue
