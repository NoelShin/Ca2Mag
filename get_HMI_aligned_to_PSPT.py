import os
from os.path import split,splitext
from csv import reader
from glob import glob
import numpy as np
from time import sleep
from shutil import copy
from tqdm import tqdm
import time

FACTOR_STD = 0

dir_PSPT = './Images/PSPT/Over_{}_std_among_whole'.format(str(FACTOR_STD))
dir_HMI = '/DATA/RAID/Noel/Datasets/HMI/Image'

# Get PSPT paths which satisfies criterion.
list_PSPT = sorted(glob(os.path.join(dir_PSPT, '*.png')))

# Get all HMI paths
list_HMI = list()
for dir in sorted(os.listdir(dir_HMI)):
    list_HMI.extend(sorted(glob(os.path.join(dir_HMI, dir, '*.png'))))

# Get all times of PSPT from the paths
list_PSPT_time = list()
list_PSPT_raw_time = list()  # This is for getting grad_sum
for PSPT in list_PSPT:
    list_PSPT_time.append(split(splitext(PSPT)[0])[-1].replace('_', ''))
    list_PSPT_raw_time.append(split(splitext(PSPT)[0])[-1])

list_grad_sum = list()
with open('./grad_sum.txt', 'r') as csv:
    csv_reader = reader(csv)
    for raw_time in list_PSPT_raw_time:
        for i, row in enumerate(csv_reader):
            if row[0] == raw_time:
                list_grad_sum.append(row[1])
                break


# Get all times of HMI from the paths
list_HMI_time = list()
for HMI in list_HMI:
    list_HMI_time.append(split(splitext(HMI)[0])[-1][11:-2].replace('_', ''))

# Get dates of PSPT from the times
list_PSPT_date = list()
for PSPT_time in list_PSPT_time:
    list_PSPT_date.append(PSPT_time[:8])
set_PSPT_date = frozenset(list_PSPT_date)

# Get dates of HMI from the times
list_HMI_date = list()
for HMI_time in list_HMI_time:
    list_HMI_date.append(HMI_time[:8])
set_HMI_date = frozenset(list_HMI_date)

# for PSPT_time in list_PSPT_time:
#     date = PSPT_time[:8]
#     for HMI_time in list_HMI_time:
#         if HMI_time[:8] == date:
#         else:
#             continue

# Get index k where PSPT is available to be paired with HMI
k = 0
for i, PSPT_date in enumerate(list_PSPT_date):
    if int(PSPT_date[:8]) >= 20110101:
        print(i)
        k = i
        break

# Get the closest time of HMI to PSPT
list_aligned_PSPT = list()
list_PSPT_index = list()
list_aligned_HMI = list()
index = 0
for PSPT_time in tqdm(list_PSPT_time[k:]):
    if int(PSPT_time[-2:]) in list(range(6, 18)):
        close_HMI_time = PSPT_time[:-2] + '12'

    elif int(PSPT_time[-2:]) in list(range(8, 30)):
        close_HMI_time = PSPT_time[:-2] + '24'

    elif int(PSPT_time[-2:]) in list(range(30, 42)):
        close_HMI_time = PSPT_time[:-2] + '36'

    elif int(PSPT_time[-2:]) in list(range(42, 54)):
        close_HMI_time = PSPT_time[:-2] + '48'

    else:
        if int(PSPT_time[-2:]) in list(range(54, 60)):
            close_HMI_time = str(int(PSPT_time) + 100)[:-2] + '00'
        else:
            close_HMI_time = PSPT_time[:-2] + '00'

    if close_HMI_time[-4:-2] == '24':
        close_HMI_time = str(int(close_HMI_time) + 10000)

    for HMI_time in list_HMI_time:
        if HMI_time == close_HMI_time:
            list_aligned_HMI.append(HMI_time)
            list_aligned_PSPT.append(PSPT_time)
            list_PSPT_index.append(k + index)
    index += 1

list_PSPT_grad_sum = list()
for index in list_PSPT_index:
    list_PSPT_grad_sum.append(list_grad_sum[index])


with open(os.path.join(dir_PSPT, 'list_aligned_HMI.txt'), 'wt') as log:
    log.write('PSPT_time, closest_HMI_time, PSPT_grad_sum\n')
    for i in range(len(list_aligned_HMI)):
        log.write(list_aligned_PSPT[i] + ', ' + list_aligned_HMI[i] + ', ' + list_PSPT_grad_sum[i] + '\n')
    log.close()


list_same_time = list()
prev_same_time = list()
for i in range(len(list_aligned_HMI) - 1):
    carrier = list()
    for j in range(len(list_aligned_HMI[i + 1:])):
        if (list_aligned_HMI[i] == list_aligned_HMI[i + j + 1]) and (list_aligned_PSPT[i + j + 1] not in prev_same_time):
            carrier.append(list_aligned_PSPT[i + j + 1])
        else:
            break
    if len(carrier) != 0:
        list_same_time += [[list_aligned_PSPT[i]] + carrier]
        prev_same_time += list(frozenset(carrier))


list_max_index = list()
for same_times in list_same_time:
    carrier = list()
    for time in same_times:
        switch = 0
        with open(os.path.join(dir_PSPT, 'list_aligned_HMI.txt'), 'r') as log:
            csv_reader = reader(log)
            for row in csv_reader:
                if time == row[0]:
                    carrier.append(int(row[2]))
                    switch = 1
                else:
                    if switch == 1:
                        break
                    else:
                        continue
            log.close()
    try:
        carrier = np.array(carrier)
        list_max_index.append(carrier.argmax())

    except ValueError:
        continue

list_max_time = list()
for i, index in enumerate(list_max_index):
    list_max_time.append(list_same_time[i][index])

list_same_time_individual = list()
for l in list_same_time:
    list_same_time_individual.extend(l)

list_delete = sorted(list(frozenset(list_same_time_individual) - frozenset(list_max_time)))


# Finally !!!!
dst = './Images/PSPT/Aligned_{}_std'.format(FACTOR_STD)
os.makedirs(dst, exist_ok=True)
for PSPT in list_PSPT:
    time = split(splitext(PSPT)[0])[-1].replace('_', '')
    if time not in list_delete:
        copy(PSPT, dst)


