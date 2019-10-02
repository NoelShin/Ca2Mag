import os
import csv
import numpy as np
from shutil import copy
from tqdm import tqdm

factor_std = 1

PSPT = './Images/PSPT'
yrs = sorted(os.listdir(PSPT))
dummy_dir = list()
for yr in yrs:
    if yr not in list(str(i) for i in range(2005, 2016)):
        dummy_dir.append(yr)

yrs = sorted(list(frozenset(yrs) - frozenset(dummy_dir)))
print(yrs)

list_file_path = list()
list_grad = list()
for yr in yrs:
    with open(os.path.join(PSPT, yr, 'Gradient/2/grads_sum.txt'), 'r') as log:
        csv_reader = csv.reader(log, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
                line_count += 1
            else:
                list_file_path.append(os.path.join(PSPT, yr, 'Raw', row[0] + '.png'))
                list_grad.append(int(row[1].strip(' ')))

array_grad = np.array(list_grad)
min, max, mean, std = array_grad.min(), array_grad.max(), array_grad.mean(), array_grad.std()
print("Total nb: ", len(array_grad), "min: ", min, "max: ", max, "mean: ", mean, "std: ", std)

dir_best = './Images/PSPT'
dst_final = './Images/PSPT/Over_{}_std_among_whole'.format(str(factor_std))
os.makedirs(dst_final, exist_ok=True)
txt_file = os.path.join(dst_final, 'Over_{}_std_among_whole.txt'.format(str(factor_std)))

criterion = mean + factor_std * std

over_criterion = list()
for i in list_grad:
    if i > criterion:
        over_criterion.append(i)

print("criterion value: {}, # of images: {}".format(criterion, len(over_criterion)))

with open(txt_file, 'wt') as log:
    pass

for i in tqdm(range(len(list_grad))):
    if list_grad[i] > criterion:
        copy(list_file_path[i], dst_final)

        with open(txt_file, 'a') as log:
            log.write(os.path.splitext(os.path.split(list_file_path[i])[-1])[0] + '\n')
