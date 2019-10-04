import os
from os.path import split, splitext
from glob import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import Compose, Crop, ToTensor, Normalize, Pad
import numpy as np
from PIL import Image
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        dataset_dir = os.path.join('./datasets', opt.dataset_name)
        format = opt.format

        if opt.is_train:
            self.label_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Input', '*.' + format)))
            self.target_path_list = sorted(glob(os.path.join(dataset_dir, 'Train', 'Target', '*.' + format)))

        elif not opt.is_train:
            self.label_path_list = sorted(glob(os.path.join(dataset_dir, 'Test', 'Input', '*.' + format)))
            self.target_path_list = sorted(glob(os.path.join(dataset_dir, 'Test', 'Target', '*.' + format)))

    def get_transform(self, normalize=True):
        transform_list = []

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        transforms = Compose(Pad((10, 10)),
                             RandomCrop((1024, 1024)),
                             ToTensor(),
                             Normalize(mean=[0.5], std=[0.5]))

        label_array = Image.open(self.label_path_list[index])
        label_tensor = transforms(label_array)

        target_array = Image.open(self.target_path_list[index])
        target_tensor = transforms(target_array)

        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
               splitext(split(self.target_path_list[index])[-1])[0]

    def __len__(self):
        return len(self.label_path_list)


if __name__ == '__main__':
    from options import TrainOption
    # from options import TestOption
    opt = TrainOption().parse()
    #test_opt = TestOption().parse()
    dataset = CustomDataset(opt)
