import os
import argparse


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=False, help='for checking code')
        self.parser.add_argument('--gpu_ids', type=int, default=1, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--HD', action='store_true', default=True, help='if True, use pix2pixHD')

        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='Over_0_std_0107', help='[Cityscapes, Custom]')
        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype')
        self.parser.add_argument('--image_height', type=int, default=1024, help='[512, 1024]')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--max_ch', type=int, default=1024, help='maximum number of channel for pix2pix')
        self.parser.add_argument('--n_downsample', type=int, default=5,
                                 help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_workers', type=int, default=2, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection',
                                 help='[reflection, replication, zero]')
        self.parser.add_argument('--use_boundary_map', action='store_true', default=True,
                                 help='if you want to use boundary map')
        self.parser.add_argument('--val_during_train', action='store_true', default=False)

    def parse(self):
        opt = self.parser.parse_args()

        opt.format = 'png'
        opt.n_df = 64
        opt.input_ch = 1
        opt.flip = False

        opt.n_gf = 32 if opt.HD and (opt.image_height == 1024) else 64
        opt.output_ch = 1

        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8

        dataset_name = opt.dataset_name
        model_name = "pix2pixHD" if opt.HD else 'pix2pix'
        model_name += "_padding" if opt.padding_size > 0 else ''
        model_name += "_rotation{}".format(str(opt.max_rotation_angle)) if opt.max_rotation_angle > 0 else ''

        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Training', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Test', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Model', model_name), exist_ok=True)

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

            return opt

        else:
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

            return opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=True, help='train flag')

        # data augmentation
        self.parser.add_argument('--padding_size', default=0, help='padding size')
        self.parser.add_argument('--max_rotation_angle', type=int, default=30, help='rotation angle(degree)')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--FM', action='store_true', default=True, help='switch for feature matching loss')
        self.parser.add_argument('--flip', action='store_true', default=True, help='switch for flip input data')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[GAN, LSGAN, WGAN_GP]')
        self.parser.add_argument('--lambda_FM', type=int, default=10, help='weight for FM loss')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=5000)
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='if you want to shuffle the order')
        self.parser.add_argument('--n_D', type=int, default=2,
                                 help='how many discriminators in differet scales you want to use')
        self.parser.add_argument('--n_epochs', type=int, default=200, help='how many epochs you want to train')
        self.parser.add_argument('--VGG_loss', action='store_true', default=True,
                                 help='if you want to use VGGNet for additional feature matching loss')
        self.parser.add_argument('--val_freq', type=int, default=5000)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=False, help='test flag')
        self.parser.add_argument('--shuffle', action='store_true', default=False,
                                 help='if you want to shuffle the order')
