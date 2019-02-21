import cv2
from glob import glob
import numpy as np
import os
#import pandas as pd
import torch
import torch.utils.data as data

from data.dstl_dataset.image_data import ImageData
from data.dstl_dataset.preprocess_utils import adjust_size, downsample, rand_rotate_and_crop
from data.dstl_dataset.dataset import normalize

_x_crop = 3336
_y_crop = 3336


_val_ids = list(range(24, 450, 25)) # val set is 18
_train_ids = list(set(range(450)) - set(_val_ids)) # train set is 432


class DstlDataset(data.Dataset):

    def __init__(self, opt):
        super(DstlDataset, self).__init__()
        # data_dir = '/Users/zhoufang/Desktop/lu/dstl'
        self.data_dir = opt['dataroot_HR']
        #train_wkt_v4 = pd.read_csv(os.path.join(self.data_dir, 'train_wkt_v4.csv'))
        #grid_sizes = pd.read_csv(os.path.join(self.data_dir, 'grid_sizes.csv'),
        #            skiprows=1, names=['ImageId', 'Xmax', 'Ymin'])

        #self.data_names = sorted(train_wkt_v4.ImageId.unique())
        all_data_names = glob(os.path.join(self.data_dir, 'three_band/*.tif'))

        self.data_names = [all_data_names[i] for i in _train_ids] \
            if opt['phase'] == 'train' else [all_data_names[i] for i in _val_ids]

        self.scale = opt['scale']
        self.patch_size = opt['HR_size']
        self.total_imgs = len(self.data_names)


    def __getitem__(self, index):
        data_id = self.data_names[index]
        image_data = ImageData(self.data_dir, data_id, grid_sizes=None, train_wkt_v4=None)
        image_data.create_train_feature()
        image = image_data.train_feature[:_y_crop, :_x_crop, :]
        image = image.astype(np.float)

        #image = adjust_size(image, self.scale)
        #image, _ = rand_rotate_and_crop(image, self.patch_size, label=None)
        image = image[:500, :500, :]

        # TODO(coufon): scale image to [0, 1].
        image = normalize(image)
        image_lr = downsample(image, self.scale)

        return {
            'LR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image_lr, (2, 0, 1))[np.newaxis, ...])).float(),
            'HR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image, (2, 0, 1)))).float()
        }


    def __len__(self):
        return self.total_imgs # number of patches
