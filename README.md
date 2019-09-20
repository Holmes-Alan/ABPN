# ABPN (Attention based Back Projection Network for image ultra-resolution)

By Zhi-Song, Li-Wen Wang and Chu-Tak Li

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a single image super-resolution using Attention based Back Projection Network (ABPN) to achieve good SR performance with low distortion.


# For proposed ABPN model, we claim the following points:

• Attention Back Projection Block to learn cross-correlation of features.

• Refined Back Projection Block to estimate super-resolution residues for better reconstruction.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch 1.1 
    NVIDIA GPU + CUDA
    MATLAB 6.0 or above

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/network.png)

# Implementation
## 1. Quick testing
---------------------------------------
### Copy your image to folder "LR" and run test.sh. The SR images will be in folder "Result"

## 2. Testing for AIM 2019
---------------------------------------
### s1. For AIM2019 Extreme Super-Resolution Challenge - Track 1: Fidelity, run **AIM_ensemble_16x.py**. Modify the directories of files based on your working environment.

Testing images on AIM2019 Extreme Super-Resolution Challenge - Track 1: Fidelity can be downloaded from the following link:

https://competitions.codalab.org/competitions/20235

### s2. For AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization, run **AIM_ensemble_4x.py**. Modify the directories of files based on your working environment.

Testing images on AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization can be downloaded from the following link:

https://competitions.codalab.org/competitions/20169

General testing dataset (Set5, Set14, BSD100, Urban100 and Manga109) can be downloaded from:

https://github.com/LimBee/NTIRE2017

## 3. Training
---------------------------
### s1. Download the training images from DIV2K and Flickr.
    
https://data.vision.ee.ethz.ch/cvl/DIV2K/

https://github.com/LimBee/NTIRE2017
   
### s2. Start training on Pytorch
For user who already has installed Pytorch 1.1, simply just run the following code for AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization:
```sh
$ python main_4x.py
```

or run the following code for AIM2019 Extreme Super-Resolution Challenge - Track 1: Fidelity:
```sh
$ python main_16x.py
```
---------------------------
  
# Experimental results

Validation results on AIM2019 Extreme Super-Resolution Challenge - Track 1: Fidelity can be downloaded from the following link:

https://drive.google.com/open?id=1rMzeN-UmWoCNKoyApEAJ7uSF5ckX34-V

Testing results on AIM2019 Extreme Super-Resolution Challenge - Track 1: Fidelity can be downloaded from the following link:

https://drive.google.com/open?id=1lJFvNKSUxg-pioKqjA39GPqDvvv3ceYj

Validation results on AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization can be downloaded from the following link:

https://drive.google.com/open?id=12gCRnI7eUhSrb5F7TN8wm4aWXcP2c6oi

Testing results on AIM2019 Constrained Super-Resolution Challenge - Track 3: Fidelity optimization can be downloaded from the following link:

https://drive.google.com/open?id=1IMSBDtfMxEkn5v6Up3MZ0n-Uk3uN0BBm

## Partial image visual comparison
## 1. Quantitative comparison
We tested on several SR approaches on several datasets for PSNR and SSIM. We have achieve comparable or even better performance.
![table](/figure/table.pdf)

## 2. Model complexity
Our proposed ABPN can use 2~3 times less parameters to achieve same PSNR performance!
![complexity](/figure/complexity.pdf)

## 3. Visualization comparison
Results on 4x image SR on Urban100 dataset
![visual](/figure/visualization.pdf)
