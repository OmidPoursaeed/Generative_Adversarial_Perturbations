import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import math

def weights_init(m, act_type='relu'):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if act_type == 'selu':
            n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
        else:
            m.weight.data.normal_(0.0, 0.02)        
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = ((0.5 ** int(epoch >= 2)) *
                    (0.5 ** int(epoch >= 5)) *
                    (0.5 ** int(epoch >= 8)))
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_iters, gamma=0.1
        )
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define(input_nc, output_nc, ngf, gen_type, norm='instance',
           act='selu', block=9, gpu_ids=[]):
    network = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if gen_type == 'unet':
        network = UnetGenerator(input_nc, output_nc, ngf, norm, act)
        network.cuda(device_id=gpu_ids[1])
    elif gen_type == 'unet-sc':
        network = UnetGeneratorSC(input_nc, output_nc, ngf, norm, act)
        network.cuda(device_id=gpu_ids[1])
    elif gen_type == 'unet-rec':
        network = RecursiveUnetGenerator(input_nc, output_nc, 8, ngf, norm, act, use_dropout=False, gpu_ids=gpu_ids)
    elif gen_type == 'resnet':
        network = ResnetGenerator(input_nc, output_nc, ngf, norm, act, use_dropout=True, n_blocks=block, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [{}] is not recognized'.format(gen_type))

    weights_init(network, act)
    return network


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UnetGenerator, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        # state size is (ngf x 8) x 8 x 16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.norm8(self.dconv3(self.act(d2))))
        # state size is (ngf x 8) x 16 x 32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.norm8(self.dconv4(self.act(d3)))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.act(d4)))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.act(d5)))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.act(d6)))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.act(d7))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 2:
            model1 = []
            mult = 2**n_downsampling
            mid = int(n_blocks / 2)
            for i in range(mid):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(n_blocks - mid):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 3:
            model1 = []
            model2 = []
            mult = 2**n_downsampling
            mid1 = int(n_blocks / 5)
            mid2 = mid1 + int((n_blocks - mid1) / 4.0 * 3)
            # mid = int(n_blocks / 2)
            for i in range(mid1):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid1, mid2):
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid2, n_blocks):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        if self.num_gpus >= 2:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model1 += [nn.ReflectionPad2d(3)]
            model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model1 += [nn.Tanh()]
        else:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model0 += [nn.ReflectionPad2d(3)]
            model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model0 += [nn.Tanh()] 

        self.model0 = nn.Sequential(*model0)
        self.model0.cuda(self.gpulist[0])
        if self.num_gpus == 2:
            self.model1 = nn.Sequential(*model1)
            self.model1.cuda(self.gpulist[1])
        if self.num_gpus == 3:
            self.model2 = nn.Sequential(*model2)
            self.model2.cuda(self.gpulist[2])

    def forward(self, input):
        input = input.cuda(self.gpulist[0])
        input = self.model0(input)
        if self.num_gpus == 3:
            input = input.cuda(self.gpulist[2])
            input = self.model2(input)
        if self.num_gpus == 2:
            input = input.cuda(self.gpulist[1])
            input = self.model1(input)
        return input


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UnetGeneratorSC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UnetGeneratorSC, self).__init__()
        self.name = 'unetsc'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv9 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv0 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.dconv1 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.dconv2 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.dconv3 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.dconv4 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1, 1)
        self.dconv5 = nn.Conv2d(ngf * 8 * 2, ngf * 4, 3, 1, 1)
        self.dconv6 = nn.Conv2d(ngf * 4 * 2, ngf * 2, 3, 1, 1)
        self.dconv7 = nn.Conv2d(ngf * 2 * 2, ngf, 3, 1, 1)
        self.dconv8 = nn.Conv2d(ngf * 2, output_nc, 3, 1, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.upsamp = nn.Upsample(scale_factor=2)
        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        e8 = self.norm8(self.conv8(self.leaky_relu(e7)))
        e9 = self.conv9(self.leaky_relu(e8))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        d0_ = self.dropout(self.norm8(self.dconv0(self.upsamp(self.act(e9)))))
        d0_ = torch.cat((d0, e8), 1)
        d1_ = self.dropout(self.norm8(self.dconv1(self.upsamp(self.act(d0)))))
        # state size is (ngf x 8) x 4 x 8
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.norm8(self.dconv2(self.upsamp(self.act(d1)))))
        # state size is (ngf x 8) x 8 x 16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.norm8(self.dconv3(self.upsamp(self.act(d2)))))
        # state size is (ngf x 8) x 16 x 32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.norm8(self.dconv4(self.upsamp(self.act(d3))))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.upsamp(self.act(d4))))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.upsamp(self.act(d5))))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.upsamp(self.act(d6))))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.upsamp(self.act(d7)))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class RecursiveUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_type,
                 act_type='selu', use_dropout=False, gpu_ids=[]):
        super(RecursiveUnetGenerator, self).__init__()
        self.name = 'unet-rec'
        self.gpu_ids = gpu_ids

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, self.act, self.gpu_ids, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(output_nc, ngf, self.act, self.gpu_ids, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, act, gpu_ids, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.gpulist = gpu_ids
        use_bias = norm_layer == 'instance'
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = act
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        if self.outermost:
            self.model0 = nn.Sequential(*down)
            self.model0.cuda(self.gpulist[0])
            self.model1 = submodule
            self.model1.cuda(self.gpulist[1])
            self.model2 = nn.Sequential(*up)
            self.model2.cuda(self.gpulist[0])
        else:
            self.model = nn.Sequential(*model)
            self.model.cuda(self.gpulist[1])

    def forward(self, x):
        if self.outermost:
            x = x.cuda(self.gpulist[0])
            x0 = self.model0(x).cuda(self.gpulist[1])
            x1 = self.model1(x0).cuda(self.gpulist[0])
            x2 = self.model2(x1)
            return x2
        else:
            return torch.cat([x, self.model(x)], 1)
        
