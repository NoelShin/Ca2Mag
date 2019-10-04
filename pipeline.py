import os
from os.path import split, splitext
from random import randint
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor, Normalize, Pad
from PIL import Image


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

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

        self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

        self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
        self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0

        transforms = Compose([Lambda(lambda x: self.__rotate(x)),
                              Pad((self.opt.padding_size, self.opt.padding_size)),
                              Lambda(lambda x: self.__random_crop(x)),
                              ToTensor(),
                              Normalize(mean=[0.5], std=[0.5])])

        label_array = Image.open(self.label_path_list[index])
        label_tensor = transforms(label_array)

        target_array = Image.open(self.target_path_list[index])
        target_tensor = transforms(target_array)

        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
               splitext(split(self.target_path_list[index])[-1])[0]

    def __random_crop(self, x):
        x = np.array(x)
        x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        return Image.fromarray(x)

    def __rotate(self, x):
        return x.rotate(self.angle)

    def __len__(self):
        return len(self.label_path_list)


if __name__ == '__main__':
    from options import TrainOption
    # from options import TestOption
    opt = TrainOption().parse()
    #test_opt = TestOption().parse()
    dataset = CustomDataset(opt)
