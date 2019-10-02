from os.path import join
import os
from glob import glob
import numpy as np

factor_std = 1
dir_PSPT = './Images/PSPT/Over_{}_std_among_whole'.format(str(factor_std))
list_PSPT = sorted(glob(join(dir_PSPT, '*.png')))

list_times = list()
list_dates = list()
for path in list_PSPT:
    name = os.path.split(os.path.splitext(path)[0])[-1]
    list_times.append(name.replace('_', ''))
    list_dates.append(name.replace('_', '')[:8])

set_dates = sorted(frozenset(list_dates))

list_same_date = list()
for date in set_dates:
    counter = 0
    carrier = []
    for _ in list_times:
        if _[:8] == date:
            carrier.append(_)
            counter = 1

        else:
            if counter == 1:
                list_same_date.append(carrier)
                break
            else:
                continue

print(list_same_date)

diff_same_date = list()
for day in list_same_date:
    carrier = list()
    if len(day) == 1:
        print(day)
    else:
        for i in range(len(day) - 1):
            carrier.append(int(day[i + 1][8:10]) * 60 + int(day[i + 1][10:12]) - int(day[i][8:10]) * 60 + int(day[i + 1][10:12]))
    diff_same_date.extend(carrier)

diff = np.array(diff_same_date).min()
print(diff)

diff_neighbor_date = list()
for i in range(len(list_same_date) - 1):
    day_0 = int(list_same_date[i][-1][6:8]) * 30 * 60 + int(list_same_date[i][-1][8:10]) * 60 + int(list_same_date[i][-1][10:12])
    day_1 = int(list_same_date[i + 1][-1][6:8]) * 30 * 60 + int(list_same_date[i + 1][-1][8:10]) * 60 + int(list_same_date[i + 1][-1][10:12])
    print(day_0, day_1)
    diff_neighbor_date.append(day_1 - day_0)

print(np.array(diff_neighbor_date).min())
