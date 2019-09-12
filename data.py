from os.path import join
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from dataset import DatasetFromFolderEval, DatasetFromFolder
from data_transform import *

def transform():
    return Compose([
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# def label_transform():
#     return Compose([
#         ToLabel(),
#         Relabel(255, 255),
#     ])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, 'HR')
    lr_dir = join(data_dir, 'LR_16x')
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                             transform=transform())

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, upscale_factor,
                             transform=transform())

