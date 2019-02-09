import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.utils.data as data

from data.dstl_dataset.image_data import ImageData
from data.dstl_dataset.preprocess_utils import adjust_size, downsample, rand_rotate_and_crop


class DstlDataset(data.Dataset):

    def __init__(self, opt):
        super(DstlDataset, self).__init__()
        # data_dir = '/Users/zhoufang/Desktop/lu/dstl'
        data_dir = opt['dataroot_HR']
        train_wkt_v4 = pd.read_csv(os.path.join(data_dir, 'train_wkt_v4.csv'))
        grid_sizes = pd.read_csv(os.path.join(data_dir, 'grid_sizes.csv'),
                    skiprows=1, names=['ImageId', 'Xmax', 'Ymin'])

        train_names = sorted(train_wkt_v4.ImageId.unique())

        x_crop = 3345
        y_crop = 3338
        self.scale = opt['scale']
        self.patch_size = opt['HR_size']
        self.images = []
        self.labels = []

        for img_id in train_names:
            image_data = ImageData(data_dir, img_id, grid_sizes, train_wkt_v4)
            image_data.create_train_feature()
            image_data.create_label()

            #cv2.imshow('image', (image_data.train_feature[1000:2000, 1000:2000]/3.0).astype(np.uint8))
            #cv2.waitKey(0)
            #plt.imshow((image_data.train_feature/3).astype(np.uint8))
            #plt.show()
            #break

            # Reduce the dimension of label classes.
            image_data.label[:, :, 0] = image_data.label[:, :, 0] + image_data.label[:, :, 1]
            image_data.label[:, :, 1] = image_data.label[:, :, 2] + image_data.label[:, :, 3]
            image_data.label[:, :, 2] = image_data.label[:, :, 4]
            image_data.label[:, :, 3] = image_data.label[:, :, 5]
            image_data.label[:, :, 4] = image_data.label[:, :, 6] + image_data.label[:, :, 7]
            image_data.label[:, :, 5] = image_data.label[:, :, 8] + image_data.label[:, :, 9] 

            self.images.append(image_data.train_feature[:x_crop, :y_crop, :])
            self.labels.append(image_data.label[:x_crop, :y_crop, :6])

        self.total_imgs = len(self.images)


    def __getitem__(self, index):
        rand_idx = np.random.randint(self.total_imgs)
        image = self.images[rand_idx]
        label = self.labels[rand_idx]

        #image = adjust_size(image, self.scale)
        #label = adjust_size(label, self.scale)
        image, label = rand_rotate_and_crop(image, label, self.patch_size)
        # TODO(coufon): scale image to [0, 1].
        image = image/2000.0
        image_lr = downsample(image, self.scale)

        return {
            'LR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image_lr.astype(np.float), (2, 0, 1)))).float(),
            'HR': torch.from_numpy(np.ascontiguousarray(
                np.transpose(image.astype(np.float), (2, 0, 1)))).float(),
            'seg': torch.from_numpy(np.ascontiguousarray(
                np.transpose(label.astype(np.float), (2, 0, 1)))).float()
        }


    def __len__(self):
        return self.total_imgs * 538 # number of patches
