from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import ABPN_v5
from collections import OrderedDict
from data import get_training_set
import numpy as np
import socket
import time
from torch.distributions.normal import Normal
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/data1/DIV2K+Flickr')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='VAE')
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped LR image')
parser.add_argument('--pretrained_sr', default='ABPN_epoch_250.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='Model/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='logs/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch, writer):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input)

        # Reconstruction loss
        loss = L1_criterion(prediction, target)
        #loss_feat = L1_criterion(SR_feat, HR_feat) / HR_feat.size(0)
        #loss = loss_recons + kl_div


        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss.item(), epoch, iteration)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.data,
                                                                                 (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder + opt.model_type + "_epoch_{}.pth".format(
        epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.upscale_factor, opt.patch_size,
                             opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)


model = ABPN_v5(input_dim=3, dim=32)
#model = HBPN_v1(input_dim=3, dim=64)
#model = torch.nn.DataParallel(model)
L1_criterion = nn.L1Loss()
L2_criterion = nn.MSELoss(size_average=False)
#Softmax_criterion = nn.CrossEntropyLoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        # print('Pre-trained SR model is loaded.')

        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

if cuda:
    model = model.cuda(gpus_list[0])
    L1_criterion = L1_criterion.cuda(gpus_list[0])
    L2_criterion = L2_criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

writer = SummaryWriter('./{0}'.format(opt.log_folder))
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch, writer)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)
