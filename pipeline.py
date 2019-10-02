import os
from os.path import split, splitext
from glob import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Pad
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

    def get_transform(self, normalize=True, label=True):
        transform_list = []

        if self.opt.is_train and self.coin:
            transform_list.append(transforms.Lambda(lambda x: self.__flip(x)))

        transform_list.append(transforms.ToTensor())

        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        return transforms.Compose(transform_list)

    @staticmethod
    def get_edges(instance_tensor):
        edge = torch.ByteTensor(instance_tensor.shape).zero_()
        edge[:, :, 1:] = edge[:, :, 1:] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, :, :-1] = edge[:, :, :-1] | (instance_tensor[:, :, 1:] != instance_tensor[:, :, :-1])
        edge[:, 1:, :] = edge[:, 1:, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])
        edge[:, :-1, :] = edge[:, :-1, :] | (instance_tensor[:, 1:, :] != instance_tensor[:, :-1, :])

        return edge.float()

    @staticmethod
    def __flip(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    def encode_input(self, label_tensor):
        return label_tensor

    def __getitem__(self, index):
        self.coin = None

        label_array = Image.open(self.label_path_list[index])
        label_tensor = self.get_transform()(label_array)

        target_array = Image.open(self.target_path_list[index])
        target_tensor = self.get_transform(label=False)(target_array)

        input_tensor = self.encode_input(label_tensor)

        return input_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
               splitext(split(self.target_path_list[index])[-1])[0]

    def __len__(self):
        return len(self.label_path_list)


if __name__ == '__main__':
    from options import TrainOption
    # from options import TestOption
    opt = TrainOption().parse()
    #test_opt = TestOption().parse()
    dataset = CustomDataset(opt)
