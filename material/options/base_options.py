import argparse
import os
from ..utils import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options')
        self.initalized = False

    def initialize(self):
        self.parser.add_argument(
            '--dataroot', help='path of the experiment data',
            default='~/data/datasets'
        )
        self.parser.add_argument(
            '--dataset', help='the dataset you use',
            type=str, required=True
        )
        self.parser.add_argument(
            '--input_nc', help='# of input channels',
            type=int, default=3
        )
        self.parser.add_argument(
            '--output_nc', help='# of output channels',
            type=int, default=3
        )
        self.parser.add_argument(
            '--ngf', help='# of generator filters in first conv layer',
            type=int, default=128
        )
        self.parser.add_argument(
            '-g', '--generator', help='type of generator',
            type=str, default='unet', required=True
        )
        self.parser.add_argument(
            '--gpu_ids', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
            type=str, default='0'
        )
        self.parser.add_argument(
            '--eps', help='episilon of perturbation',
            type=int, default=10, required=True
        )
        self.parser.add_argument(
            '--name', help='name of experimentz: decide the where log files and models stored',
            type=str, required=True
        )
        self.parser.add_argument(
            '--norm', help='type of normalization',
            type=str, default='instance'
        )
        self.parser.add_argument(
            '--activation', help='type of activation function',
            type=str, default='selu'
        )
        self.parser.add_argument(
            '--nThreads', default=4,
            type=int, help='# of threads for loading data'
        )
        self.parser.add_argument(
            '-r', '--resolution', help='resolution of dataset',
            type=int, default=1024
        )
        self.parser.add_argument(
            '--val-interval', help='interval size of validation',
            type=int, default=1500
        )
        self.parser.add_argument(
            '--batch-size', help='size of each batch',
            type=int, default=1
        )
        self.parser.add_argument(
            '--block', help='number of blocks in resnet generator',
            type=int, default=9
        )
        self.parser.add_argument(
            '--checkpoints_dir', type=str,
            default='./checkpoints', help='models are saved here'
        )
        self.parser.add_argument(
            '--seed', type=int,
            default='123', help='random seed'
        )
        self.parser.add_argument(
            '--target_path', type=str,
            default='~/data/datasets/cityscapes-{}/gtFine/train/monchengladbach/monchengladbach_000000_026602.png',
            help='path of the target mask'
        )
        self.parser.add_argument(
            '--pretrained_cityscapes', type=str,
            default='~/data/models/pytorch/model_best.pth.tar',
            help='path of pretrained model based on cityscapes dataset'
        )
        self.parser.add_argument(
            '--pretrained_pascalvoc', type=str,
            default='~/data/models/pytorch/fcn8s_from_caffe.pth',
            help='path of pretrained model based on pascalvoc dataset'
        )
        self.parser.add_argument(
            '--metric', type=int,
            default=0, required=True,
            help='metrics showed in visdom, like success_rate or mean_iu'
        )
        self.parser.add_argument(
            '--task', type=int,
            default=3, required=True,
            help='task we need to do'
        )
        self.parser.add_argument(
            '--alpha', type=float,
            default=1.0,
            help='alpha parameter'
        )
        self.parser.add_argument(
            '--resume', type=bool,
            default=False,
            help='indicator of whether you want to resume from a exisiting model'
        )
        self.parser.add_argument(
            '--resume_name', type=str,
            help='the directory of the model you want to resume'
        )
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_window_size', type=int, default=256, help='display window size')
        self.parser.add_argument('--test', type=bool, default=False, help='enable test mode or not')
        self.parser.add_argument('--save', type=bool, default=False, help='enable save mode or not')
        self.initalized = True

    def parse(self):
        if not self.initalized:
            self.initialize()
        self.args = self.parser.parse_args()
        self.args.isTrain = self.isTrain  # Train mode or Test mode
        self.args.pretrained_cityscapes = os.path.expanduser(self.args.pretrained_cityscapes)
        self.args.pretrained_pascalvoc = os.path.expanduser(self.args.pretrained_pascalvoc)
        self.args.target_path = os.path.expanduser(self.args.target_path)
        # conver the gpu_ids from string to list
        gpu_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for gpu_id in gpu_ids:
            id = int(gpu_id)
            if id >= 0:
                self.args.gpu_ids.append(id)
        self.args.gpu_ids.sort()

        # print the arguments
        args = vars(self.args)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # # save to the disk
        expr_dir = os.path.join(self.args.checkpoints_dir, self.args.name)
        util.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.args
