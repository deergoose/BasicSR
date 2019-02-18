import numpy as np
import matplotlib.pyplot as plt
import random
import tifffile

from data_utils import *


class ImageData():

    def __init__(self, data_dir, image_id, grid_sizes=None, train_wkt_v4=None, phase='train'):
        self.data_dir = data_dir
        self.image_id = image_id
        self.grid_sizes = grid_sizes
        self.train_wkt_v4 = train_wkt_v4
        self.three_band_image = None
        self.sixteen_band_image = None
        self.image = None
        self.image_size = None
        self._xymax = None
        self.label = None
        self.crop_image = None
        self.train_feature = None
        self.pred_mask = None


    def load_pre_mask(self):
        self.pred_mask = None


    def load_image(self):
        '''
        Load three band and sixteen band images, registered and at the same
        resolution
        Assign value for image_size
        :return:
        '''
        # Read data in format (c, w, h)
        im3 = tifffile.imread(self.image_id)
        #[nx, ny, _] = im3.shape
        #ima = resize(np.transpose(
        #    tifffile.imread('{}/sixteen_band/{}_A.tif'.format(self.data_dir, self.image_id)),
        #        (1, 2, 0)), [nx, ny])
        #imm = resize(np.transpose(
        #    tifffile.imread('{}/sixteen_band/{}_M.tif'.format(self.data_dir, self.image_id)),
        #        (1, 2, 0)), [nx, ny])
        #imp = np.expand_dims(resize(tifffile.imread(
        #    '{}/sixteen_band/{}_P.tif'.format(self.data_dir, self.image_id)), [nx, ny]), 2)
        #warp_matrix_a = np.load(
        #    (data_dir + '/utils/image_alignment/{}_warp_matrix_a.npz').format(self.image_id))
        #warp_matrix_m = np.load(
        #    (data_dir + '/utils/image_alignment/{}_warp_matrix_m.npz').format(self.image_id))
        #ima = affine_transform(ima, warp_matrix_a, [nx, ny])
        #imm = affine_transform(imm, warp_matrix_m, [nx, ny])
        #self.image = im3
        #self.image = np.concatenate((im3, ima, imm, imp), axis=-1)
        self.three_band_image = im3
        #self.sixteen_band_image = self.image[..., 3:]
        self.image_size = np.shape(self.image)[1:3]
        self.image = self.three_band_image

        #if self.grid_sizes is not None:
        #    xmax = self.grid_sizes[self.grid_sizes.ImageId==self.image_id].Xmax.values[0]
        #    ymax = self.grid_sizes[self.grid_sizes.ImageId==self.image_id].Ymin.values[0]
        #    self._xymax = [xmax, ymax]


    def create_label(self):
        '''
        Create the class labels
        :return:
        '''
        if self.image is None:
            self.load_image()
        labels = np.zeros(np.append(self.image_size, len(CLASSES)), np.uint8)

        for cl in CLASSES:
            polygon_list = get_polygon_list(self.image_id, cl, self.train_wkt_v4)
            perim_list, inter_list = generate_contours(
                polygon_list, self.image_size, self._xymax)
            mask = generate_mask_from_contours(
                self.image_size, perim_list, inter_list, class_id=1)
            labels[..., cl - 1] = mask
        self.label = labels


    def create_train_feature(self):
        '''
        Create synthesized features
        :return:
        '''
        if self.three_band_image is None:
            self.load_image()

        #m = self.sixteen_band_image[..., 8:].astype(np.float32)
        rgb = self.three_band_image #.astype(np.float32)

        #feature = np.concatenate([m, rgb], 2)
        #feature[feature == np.inf] = 0
        #feature[feature == -np.inf] = 0

        self.train_feature = rgb
