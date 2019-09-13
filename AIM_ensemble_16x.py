from __future__ import print_function
import argparse

import os
import torch
from model import ABPN_v3
import torchvision.transforms as transforms
from collections import OrderedDict
import logging
import numpy as np
from os.path import join
import time
import math
from dataset import is_image_file
import utils_logger
from PIL import Image, ImageOps
from os import listdir
from prepare_images import *
import torch.utils.data as utils
from torch.autograd import Variable
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=16, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=64, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='Image/DIV8K_test')
parser.add_argument('--model_type', type=str, default='ABPN')
parser.add_argument('--image_output', default='Image/DIV8K_test', help='Location to save checkpoint models')
parser.add_argument('--model', default='Model/ABPN_16x.pth', help='sr pretrained base model')

opt = parser.parse_args()

torch.cuda.current_device()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('===> Building model ', opt.model_type)
model = ABPN_v3(input_dim=3, dim=32)
model = model.to(device)
model = torch.nn.DataParallel(model)

model_name = os.path.join(opt.model)
if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')


img_splitter = ImageSplitter(opt.patch_size, opt.upscale_factor, opt.stride)

def eval():
    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    LR_filename = os.path.join(opt.image_dataset, 'LR')
    LR_image = [join(LR_filename, x) for x in listdir(LR_filename) if is_image_file(x)]
    SR_image = [join(os.path.join(opt.image_dataset, 'SR'), x) for x in listdir(LR_filename) if is_image_file(x)]

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(opt.image_dataset)
    logger.info(opt.image_output)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(LR_image.__len__()):

        idx += 1
        img_name, ext = os.path.splitext(LR_image[i])
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        LR = Image.open(LR_image[i]).convert('RGB')
        LR_90 = LR.transpose(Image.ROTATE_90)
        LR_180 = LR.transpose(Image.ROTATE_180)
        LR_270 = LR.transpose(Image.ROTATE_270)
        LR_f = LR.transpose(Image.FLIP_LEFT_RIGHT)
        LR_90f = LR_90.transpose(Image.FLIP_LEFT_RIGHT)
        LR_180f = LR_180.transpose(Image.FLIP_LEFT_RIGHT)
        LR_270f = LR_270.transpose(Image.FLIP_LEFT_RIGHT)

        with torch.no_grad():
            pred, time = chop_forward(LR, model, start, end)
            pred_90, time_90 = chop_forward(LR_90, model, start, end)
            pred_180, time_180 = chop_forward(LR_180, model, start, end)
            pred_270, time_270 = chop_forward(LR_270, model, start, end)
            pred_f, time_f = chop_forward(LR_f, model, start, end)
            pred_90f, time_90f = chop_forward(LR_90f, model, start, end)
            pred_180f, time_180f = chop_forward(LR_180f, model, start, end)
            pred_270f, time_270f = chop_forward(LR_270f, model, start, end)

        compute_time = time + time_90 + time_180 + time_270 + time_f + time_90f + time_180f + time_270f
        test_results['runtime'].append(compute_time)  # milliseconds
        pred_90 = np.rot90(pred_90, 3)
        pred_180 = np.rot90(pred_180, 2)
        pred_270 = np.rot90(pred_270, 1)
        pred_f = np.fliplr(pred_f)
        pred_90f = np.rot90(np.fliplr(pred_90f), 3)
        pred_180f = np.rot90(np.fliplr(pred_180f), 2)
        pred_270f = np.rot90(np.fliplr(pred_270f), 1)
        prediction = (pred + pred_90 + pred_180 + pred_270 + pred_f + pred_90f + pred_180f + pred_270f) * 255.0 / 8.0
        prediction = prediction.clip(0, 255)

        Image.fromarray(np.uint8(prediction)).save(SR_image[i])

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(opt.image_dataset, ave_runtime))

    # print("PSNR_predicted=", avg_psnr_predicted / count)

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


def chop_forward(img, network, start, end):

    channel_swap = (1, 2, 0)
    run_time = 0
    img = transform(img).unsqueeze(0)
    img_patch = img_splitter.split_img_tensor(img)

    testset = utils.TensorDataset(img_patch)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)
    out_box = []

    for iteration, batch in enumerate(test_dataloader, 1):
        input = Variable(batch[0]).to(device)

        start.record()
        with torch.no_grad():
            prediction = network(input)
        end.record()
        torch.cuda.synchronize()
        run_time += start.elapsed_time(end)

        for j in range(prediction.shape[0]):
            out_box.append(prediction[j,:,:,:])

    SR = img_splitter.merge_img_tensor(out_box)
    SR = SR.data[0].numpy().transpose(channel_swap)

    return SR, run_time



##Eval Start!!!!
eval()
