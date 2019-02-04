import numpy as np
import matplotlib.pyplot as plt
import random
import tifffile

from data_utils import *


class ImageData():

    def __init__(self, data_dir, image_id, grid_sizes, train_wkt_v4, phase = 'train'):
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
        im3 = np.transpose(tifffile.imread(
            '{}/three_band/{}.tif'.format(self.data_dir, self.image_id)), (1, 2, 0))
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
        self.image = im3
        #self.image = np.concatenate((im3, ima, imm, imp), axis=-1)
        self.three_band_image = self.image[..., :3]
        self.sixteen_band_image = self.image[..., 3:]
        self.image_size = np.shape(self.image)[0:2]

        xmax = self.grid_sizes[self.grid_sizes.ImageId==self.image_id].Xmax.values[0]
        ymax = self.grid_sizes[self.grid_sizes.ImageId==self.image_id].Ymin.values[0]
        self._xymax = [xmax, ymax]


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


    def visualize_image(self, plot_all = True):
        '''
        Visualize all images and class labels
        :param plot_all:
        :return:
        '''
        if self.label is None:
            self.create_label()

        if not plot_all:
            fig, axarr = plt.subplots(figsize = [10, 10])
            ax = axarr
        else:
            fig, axarr = plt.subplots(figsize = [20, 20], ncols = 3, nrows = 3)
            ax = axarr[0][0]

        polygon_list = {}
        for cl in CLASSES:
            polygon_list[cl] = get_polygon_list(self.image_id, cl)
            print '{}: {} \t\tcount = {}'.format(
                cl, CLASSES[cl], len(polygon_list[cl]))

        legend = plot_polygon(polygon_list = polygon_list, ax = ax)

        ax.set_xlim(0, self._xymax[0])
        ax.set_ylim(self._xymax[1], 0)
        ax.set_xlabel(self.image_size[0])
        ax.set_ylabel(self.image_size[1])

        if plot_all:
            three_band_rescale = scale_percentile(self.three_band_image)
            sixteen_band_rescale = scale_percentile(self.sixteen_band_image)
            plot_image(three_band_rescale, axarr[0][1], self.image_id, '3')
            plot_overlay(three_band_rescale, axarr[0][2], self.image_id, '3',
                         polygon_list,
                         scaler = self.image_size / np.array([self._xymax[1],
                                                              self._xymax[0]]))
            axarr[0][2].set_ylim(self.image_size[0], 0)
            axarr[0][2].set_xlim(0, self.image_size[1])
            plot_image(sixteen_band_rescale, axarr[1][0], self.image_id, 'A',
                       selected_channel = [0, 3, 6])
            plot_image(sixteen_band_rescale, axarr[1][1], self.image_id, 'A',
                       selected_channel = [1, 4, 7])
            plot_image(sixteen_band_rescale, axarr[1][2], self.image_id, 'A',
                       selected_channel = [2, 5, 0])
            plot_image(sixteen_band_rescale, axarr[2][0], self.image_id, 'M',
                       selected_channel = [8, 11, 14])
            plot_image(sixteen_band_rescale, axarr[2][1], self.image_id, 'M',
                       selected_channel = [9, 12, 15])
            plot_image(sixteen_band_rescale, axarr[2][2], self.image_id, 'M',
                       selected_channel = [10, 13, 8])

        ax.legend(handles = legend,
                  bbox_to_anchor = (0.9, 0.95),
                  bbox_transform = plt.gcf().transFigure,
                  ncol = 5,
                  fontsize = 'large',
                  title = 'Objects-' + self.image_id,
                  framealpha = 0.3)


    def visualize_label(self, x_range = None, y_range = None, alpha = 1.0):
        '''
        Visualize labels
        :param plot_all:
        :return:
        '''
        if self.label is None:
            self.create_label()

        if not x_range:
            x_range = [0, self.image_size[0]]
        if not y_range:
            y_range = [0, self.image_size[1]]

        fig, ax= plt.subplots(figsize = [10, 10])

        polygon_list = {}
        for cl in CLASSES:
            polygon_list[cl] = get_polygon_list(self.image_id, cl)
            print '{}: {} \t\tcount = {}'.format(
                cl, CLASSES[cl], len(polygon_list[cl]))

        three_band_rescale = scale_percentile(self.three_band_image)
        legend = plot_overlay(
            three_band_rescale, ax, self.image_id, 'P',polygon_list,
            scaler=self.image_size / np.array([self._xymax[1], self._xymax[0]]),
            alpha = alpha, rgb = True)

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_xlabel(x_range[1] - x_range[0])
        ax.set_ylabel(x_range[1] - x_range[0])

        ax.legend(handles = legend,
                  bbox_to_anchor = (0.95, 0.99),
                  bbox_transform = plt.gcf().transFigure,
                  ncol = 5,
                  fontsize = 'large',
                  title = 'Objects-' + self.image_id,
                  framealpha = 0.3)


    def apply_crop(self, patch_size, ref_point = [0, 0], method = 'random'):

        if self.image is None:
            self.load_image()

        crop_area = np.zeros([2, 2])
        width = self.image_size[0]
        height = self.image_size[1]

        assert width >= patch_size > 0 and patch_size <= height

        if method == 'random':
            ref_point[0] = random.randint(0, width - patch_size)
            ref_point[1] = random.randint(0, height - patch_size)
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        elif method == 'grid':
            assert width > ref_point[0] + patch_size
            assert height > ref_point[1] + patch_size
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        else:
            raise NotImplementedError(
                '"method" should either be "random" or "grid"')
        self.crop_image = crop(self.image, crop_area)
