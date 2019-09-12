from __future__ import print_function
import argparse

import os
import torch
from model import HBPN_v3
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
from os.path import join
import time
import math
from dataset import is_image_file
from functools import reduce
from collections import OrderedDict
#import cv2
from PIL import Image, ImageOps
from os import listdir
from prepare_images import *
import torch.utils.data as utils
from torch.autograd import Variable
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=6, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=32, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=32, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='Image/Urban100')
parser.add_argument('--model_type', type=str, default='VAE')
parser.add_argument('--output', default='Image/Urban100', help='Location to save checkpoint models')
parser.add_argument('--model', default='models/VAE_epoch_480.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cudnn.benchmark = True
torch.cuda.empty_cache()
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)
if opt.model_type == 'VAE':
    model = HBPN_v3(input_dim=3, dim=32)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
if cuda:
    model = model.cuda(gpus_list[0])

#model_name = os.path.join(opt.model)
#new_state_dict = OrderedDict()
#state_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
#for k, v in state_dict.items():
#    name = k[7:]
#    new_state_dict[name] = v

#model.load_state_dict(new_state_dict)

print('===> Loading datasets')

img_splitter = ImageSplitter(opt.patch_size, opt.upscale_factor, opt.stride)

def eval(i):
    model_name = 'models/VAE_epoch_'+str(i)+'.pth'
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    print(model_name)
    #new_state_dict = OrderedDict()
    #state_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
    #for k, v in state_dict.items():
    #    name = k[7:]
    #    new_state_dict[name] = v

    #model.load_state_dict(new_state_dict)
    model.eval()
    LR_filename = os.path.join(opt.image_dataset, 'LR')
    HR_filename = os.path.join(opt.image_dataset, 'HR')
    LR_image = [join(LR_filename, x) for x in listdir(LR_filename) if is_image_file(x)]
    HR_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    SR_image = [join(os.path.join(opt.image_dataset, 'SR'), x) for x in listdir(HR_filename) if is_image_file(x)]
    count = 0
    avg_psnr_predicted = 0.0

    for i in range(LR_image.__len__()):
        t0 = time.time()
        target = Image.open(HR_image[i]).convert('RGB')
        LR = Image.open(LR_image[i]).convert('RGB')
        with torch.no_grad():
            prediction = chop_forward(LR, model, opt.upscale_factor, opt.stride, opt.patch_size)

        t1 = time.time()
        #print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)

        Image.fromarray(np.uint8(prediction)).save(SR_image[i])

        GT = np.array(target).astype(np.float32)
        GT_Y = rgb2ycbcr(GT)
        prediction = np.array(prediction).astype(np.float32)
        prediction_Y = rgb2ycbcr(prediction)
        psnr_predicted = PSNR(prediction_Y, GT_Y, shave_border=opt.upscale_factor)
        avg_psnr_predicted += psnr_predicted
        count += 1

    print("PSNR_predicted=", avg_psnr_predicted / count)

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    #y, cb, cr = img.split()
    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt

def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(img, network, scale, stride, patch_size):

    channel_swap = (1, 2, 0)

    img = transform(img).unsqueeze(0)
    img_patch = img_splitter.split_img_tensor(img)

    testset = utils.TensorDataset(img_patch)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)
    out_box = []

    for iteration, batch in enumerate(test_dataloader, 1):
        input = Variable(batch[0]).cuda(gpus_list[0])
        with torch.no_grad():
            prediction = network(input)

        for j in range(prediction.shape[0]):
            out_box.append(prediction[j,:,:,:])

    SR = img_splitter.merge_img_tensor(out_box)
    SR = SR.data[0].numpy().transpose(channel_swap)

    return SR



##Eval Start!!!!
for i in range (140, 315, 5):
    eval(i)
