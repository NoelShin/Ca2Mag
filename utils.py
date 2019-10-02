import os
import sys

from functools import partial
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def configure(opt):
    opt.format = 'png'
    opt.n_df = 64
    if opt.dataset_name == 'Cityscapes':
        if opt.use_boundary_map:
            opt.input_ch = 36

        else:
            opt.input_ch = 35

        if opt.image_height == 512:
            opt.half = True
        elif opt.image_height == 1024:
            opt.half = False
        opt.image_size = (512, 1024) if opt.half else (1024, 2048)
        opt.n_gf = 64 if opt.half else 32
        opt.output_ch = 3

    else:
        opt.input_ch = 1
        opt.flip = False
        if opt.image_height == 512:
            opt.half = True
        elif opt.image_height == 1024:
            opt.half = False
        opt.image_size = (512, 512) if opt.half else (1024, 1024)
        opt.n_gf = 64 if opt.half else 32
        opt.output_ch = 1

    if opt.data_type == 16:
        opt.eps = 1e-4
    elif opt.data_type == 32:
        opt.eps = 1e-8

    dataset_name = opt.dataset_name
    model_name = model_namer(height=opt.image_height, val=opt.val_during_train)
    make_dir(dataset_name, model_name, type='checkpoints')

    if opt.is_train:
        opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Training', model_name)
    else:
        opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Test', model_name)

    opt.model_dir = os.path.join('./checkpoints', dataset_name, 'Model', model_name)
    log_path = os.path.join('./checkpoints/', dataset_name, 'Model', model_name, 'opt.txt')

    if os.path.isfile(log_path) and opt.is_train:
        permission = input(
            "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(model_name + '/opt'))
        if permission == 'Y':
            pass
        else:
            raise NotImplementedError("Please check {}".format(log_path))

    if opt.debug:
        opt.display_freq = 1
        opt.epoch_decay = 2
        opt.n_epochs = 4
        opt.report_freq = 1
        opt.save_freq = 1

    with open(os.path.join(opt.model_dir, 'Analysis.txt'), 'wt') as analysis:
        analysis.write('Iteration, CorrCoef_TUMF, CorrCoef_1x1, CorrCoef_2x2, CorrCoef_4x4, CorrCoef_8x8, '
                       'R1_mean, R1_std, R2_mean, R2_std\n')

        analysis.close()

    args = vars(opt)
    with open(log_path, 'wt') as log:
        log.write('-' * 50 + 'Options' + '-' * 50 + '\n')
        print('-' * 50 + 'Options' + '-' * 50)
        for k, v in sorted(args.items()):
            log.write('{}: {}\n'.format(str(k), str(v)))
            print("{}: {}".format(str(k), str(v)))
        log.write('-' * 50 + 'End' + '-' * 50)
        print('-' * 50 + 'End' + '-' * 50)
        log.close()


def model_namer(**elements):
    name = ''
    for k, v in sorted(elements.items()):
        name += str(k) + '_' + str(v) + '_'
    name += 'without_error_L1'
    return name.strip('_')


def make_dir(dataset_name=None, model_name=None, type='checkpoints'):
    if type == 'checkpoints':
        assert model_name, "model_name keyword should be specified for type='checkpoints'"
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Training', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Test', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Model', model_name), exist_ok=True)

    else:
        """
        for other type of directory
        """
        pass


def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid


def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer


class Manager(object):
    def __init__(self, opt):
        self.opt = opt
        self.dtype = opt.data_type

    @staticmethod
    def report_loss(package):
        print("Epoch: {} [{:.{prec}}%] Current_step: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}".
              format(package['Epoch'], package['current_step']/package['total_step'] * 100, package['current_step'],
                     package['D_loss'], package['G_loss'], prec=4))

    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            if self.dtype == 32:
                scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                            np.float32(drange_in[1]) - np.float32(drange_in[0]))
                bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            elif self.dtype == 16:
                scale = (np.float16(drange_out[1]) - np.float16(drange_out[0])) / (
                            np.float16(drange_in[1]) - np.float16(drange_in[0]))
                bias = (np.float16(drange_out[0]) - np.float16(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def tensor2image(self, image_tensor):
        np_image = image_tensor.squeeze().cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)

    def save(self, package, image=False, model=False):
        if image:
            path_real = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'real.png')
            path_fake = os.path.join(self.opt.image_dir, str(package['Epoch']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D.pt')
            path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G.pt')
            torch.save(package['D_state_dict'], path_D)
            torch.save(package['G_state_dict'], path_G)

    def __call__(self, package):
        if package['current_step'] % self.opt.display_freq == 0:
            self.save(package, image=True)

        if package['current_step'] % self.opt.report_freq == 0:
            self.report_loss(package)

        if package['current_step'] % self.opt.save_freq == 0:
            self.save(package, model=True)


def update_lr(old_lr, init_lr, n_epoch_decay, D_optim, G_optim):
    delta_lr = init_lr / n_epoch_decay
    new_lr = old_lr - delta_lr

    for param_group in D_optim.param_groups:
        param_group['lr'] = new_lr

    for param_group in G_optim.param_groups:
        param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)


def binning_and_cal_pixel_cc(np_fake, np_real, bin_size, radius=392, patch=False):
    assert bin_size > 1
    dtype = np_fake.dtype
    original_h, original_w = np_fake.shape

    fake_grid = np.zeros(shape=(original_h // bin_size, original_w // bin_size), dtype=dtype)
    real_grid = np.zeros(shape=(original_h // bin_size, original_w // bin_size), dtype=dtype)

    circle_index = list()
    reduced_radius = int(radius / bin_size)
    k = 0
    for y_count, i in enumerate(range(0, original_h, bin_size)):
        for x_count, j in enumerate(range(0, original_w, bin_size)):
            fake_grid[y_count, x_count] = np_fake[i: i + bin_size, j: j + bin_size].mean()
            real_grid[y_count, x_count] = np_real[i: i + bin_size, j: j + bin_size].mean()

            if (y_count - original_h // (2 * bin_size) - 1) ** 2 + (x_count - original_w // (2 * bin_size) - 1) ** 2 <= reduced_radius ** 2:
                circle_index.append(k)
            k += 1

    fake_grid, real_grid = fake_grid.flatten(), real_grid.flatten()

    if not patch:
        carrier_bin_fake, carrier_bin_real = list(), list()
        carrier_bin_fake.append(fake_grid[circle_index])
        carrier_bin_real.append(real_grid[circle_index])

        corr = np.corrcoef(carrier_bin_fake, carrier_bin_real)[0, 1]

    else:
        corr = np.corrcoef(fake_grid, real_grid)[0, 1]
    return corr
