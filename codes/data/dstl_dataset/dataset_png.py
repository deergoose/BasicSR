import cv2
from glob import glob
import numpy as np
import os
#import pandas as pd
import torch
import torch.utils.data as data

from data.dstl_dataset.preprocess_utils import adjust_size, downsample, rand_rotate_and_crop


_val_ids = list(range(9, 900, 10)) # val set is 90
_train_ids = list(set(range(900)) - set(_val_ids)) # train set is 810

# Mean on sen2 images.
pixel_mean = [85.772, 78.706, 63.223]
pixel_max = 255.0

def normalize(im):
    for i in range(3):
        im[..., i] -= pixel_mean[i]
    im /= pixel_max
    return im

def denormalize(im):
    im = np.transpose(im, (1, 2, 0))
    im *= pixel_max
    for i in range(3):
        im[..., i] += pixel_mean[i]
    return im.astype(np.uint8)


class DstlDataset(data.Dataset):

    def __init__(self, opt):
        super(DstlDataset, self).__init__()
        self.data_dir = opt['dataroot_HR']
        all_data_names = glob(os.path.join(self.data_dir, '*.png'))
        all_data_names.sort()
        self.data_names = [all_data_names[i] for i in _train_ids] \
            if opt['phase'] == 'train' else [all_data_names[i] for i in _val_ids]

        self.scale = opt['scale']
        self.patch_size = opt['HR_size']
        self.total_imgs = len(self.data_names)
        self.cached_image_data = None
        self.counter = 0

    def __getitem__(self, index):
        # Try to use cache.
        if self.cached_image_data is not None:
            image_data = self.cached_image_data
            self.counter -= 1
            if self.counter == 0:
                self.cached_image_data = None
        else:
            np.random.seed(index)
            img_path = self.data_names[np.random.randint(self.total_imgs)]
            image_data = cv2.imread(img_path)
            self.cached_image_data = image_data
            self.counter == 2

        image = image_data.astype(np.float)

        #image = adjust_size(image, self.scale)
        image, _ = rand_rotate_and_crop(image, self.patch_size, label=None)

        # TODO(coufon): scale image to [-1, 1].
        image = normalize(image)
        image_lr = downsample(image, self.scale)

        return {
            'LR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image_lr, (2, 0, 1)))).float(),
            'HR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image, (2, 0, 1)))).float()
        }

    def __len__(self):
        return self.total_imgs * 100 # number of patches
