__author__ = 'rogerjiang'

'''
Purposes:
1. Visualization of training data
2. Evaluation of training data augmentation

'''
'''
Notes on the data files:

train_wkt_v4.csv: training labels with ImageId, ClassType, MultipolygonWKT

train_geoson_v3 (similar to train_wkt_v4.csv): training labels with ImageId 
(folder name),  ClassType (detailed, name of .geojson files), Multipolygon 
(data of .geojson files, also contains detailed ClassType information)

grid_size.csv: sizes of all images with ImageId, 0<Xmax<1, -1<Ymin<0 
(size of images, assuming origin (0,0) is at the upper left corner)

three_band: all 3-band images, in name of ImageId.tif

sixteen_band: all 16-band images, in name of ImageId_{A,M,P}.tif

sample_submission.csv: submission with ImageId, ClassType, MultipolygonWKT

If the order of dimension in all the image data is x-y, this order is switched 
to y-x in grid_sizes and wkt data from train_wkt_v4.

-------------
'''

'''
Basically, the combination of ClassType and MultipolygonWKT gives the voxel-wise 
class labels.

The 'three_band' and 'sixteen_band' folders are the input for training.

ImageId connects the class labels with the training data.

MultipolygonWKT is relative position in the figure and can be converted to pixel 
coordinate with the grid_size (Xmax, Ymin)

There is slightly mismatch between the three_band and sixteen_band data due to 
delay in measurements, such that they should be aligned.

'''
import shapely.wkt as wkt
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from matplotlib.patches import Patch
from matplotlib import cm
from shapely import affinity
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
import sys
import seaborn as sns
import os


CLASSES = {
    1: 'Bldg',
    2: 'Struct',
    3: 'Road',
    4: 'Track',
    5: 'Trees',
    6: 'Crops',
    7: 'Fast H2O',
    8: 'Slow H2O',
    9: 'Truck',
    10: 'Car',
}

COLORS = {
    1: '0.7',
    2: '0.4',
    3: '#b35806',
    4: '#dfc27d',
    5: '#1b7837',
    6: '#a6dba0',
    7: '#74add1',
    8: '#4575b4',
    9: '#f46d43',
    10: '#d73027',
}

# ZORDER defines the priority for plotting overlay of class labels.
ZORDER = {
  1: 6,
  2: 5,
  3: 4,
  4: 1,
  5: 3,
  6: 2,
  7: 7,
  8: 8,
  9: 9,
  10: 10,
}


def resize(im, shape_out):
    return cv2.resize(im, (shape_out[1], shape_out[0]),
                      interpolation=cv2.INTER_CUBIC)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def crop(img, crop_coord):
    width, height = img.shape[0], img.shape[1]
    x_lim = crop_coord[0].astype(np.int)
    y_lim = crop_coord[1].astype(np.int)
    assert 0 <= x_lim[0] < x_lim[1] <= width
    assert 0 <= y_lim[0] < y_lim[1] <= height
    return img[x_lim[0]: x_lim[1], y_lim[0]: y_lim[1]]


def affine_transform(img, warp_matrix, out_shape):
    '''
    Apply affine transformation using warp_matrix to img, and perform
    interpolation as needed
    :param img:
    :param warp_matrix:
    :param out_shape:
    :return:
    '''
    new_img = cv2.warpAffine(img, warp_matrix, (out_shape[1], out_shape[0]),
                             flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode= cv2.BORDER_REPLICATE)
    # new_img[new_img == 0] = np.average(new_img)
    return new_img


def get_polygon_list(image_id, class_type, train_wkt_v4):
    '''
    Load the wkt data (relative coordiantes of polygons) from csv file and
    returns a list of polygons (in the format of shapely multipolygon)
    :param image_id:
    :param class_type:
    :return:
    '''
    all_polygon = train_wkt_v4[train_wkt_v4.ImageId == image_id]
    polygon = all_polygon[all_polygon.ClassType == class_type].MultipolygonWKT
    # For empty polygon, polygon is a string of 'MULTIPOLYGON EMPTY'
    # wkt.loads will automatically handle this and len(polygon_list) returns 0
    # But polygon_list will never be None!
    polygon_list = wkt.loads(polygon.values[0])
    return polygon_list


def convert_coordinate_to_raster(coords, img_size, xymax):
    '''
    Converts the relative coordinates of contours into raster coordinates.
    :param coords:
    :param img_size:
    :param xymax:
    :return:
    '''
    xmax, ymax = xymax
    width, height = img_size

    coords[:, 0] *= (height + 1) / xmax
    coords[:, 1] *= (width + 1) / ymax

    coords = np.round(coords).astype(np.int32)

    return coords


def generate_contours(polygon_list, img_size, xymax):
    '''
    Convert shapely MultipolygonWKT type of data (relative coordinate) into
    list type of date for polygon raster coordinates
    :param polygon_list:
    :param img_size:
    :param xymax:
    :return:
    '''
    if len(polygon_list) == 0:
        return [], []

    to_ind = lambda x: np.array(list(x)).astype(np.float32)

    perim_list = [convert_coordinate_to_raster(to_ind(poly.exterior.coords),
                                               img_size, xymax)
                  for poly in polygon_list]
    inter_list = [convert_coordinate_to_raster(
        to_ind(poly.coords), img_size, xymax)
        for poly_ex in polygon_list for poly in poly_ex.interiors]

    return perim_list, inter_list


def generate_mask_from_contours(img_size, perim_list, inter_list, class_id = 1):
    '''
    Create pixel-wise mask from contours from polygon of raster coordinates
    :param img_size:
    :param perim_list:
    :param inter_list:
    :param class_id:
    :return:
    '''
    mask = np.zeros(img_size, np.uint8)

    if perim_list is None:
        return mask
    # mask should match the dimension of image
    # however, cv2.fillpoly assumes the x and y axes are oppsite between mask and
    # perim_list (inter_list)
    cv2.fillPoly(mask, perim_list, class_id)
    cv2.fillPoly(mask, inter_list, 0)

    return mask


def plot_polygon(polygon_list, ax, scaler = None, alpha = 0.7):
    '''
    polygon_list is a dictionary of polygon list for all class types.
    key is the class id, and value is the polygon list.
    :param polygon_list:
    :param ax:
    :param scaler:
    :param alpha:
    :return:
    '''
    legend_list = []
    for cl in CLASSES:
        # Patch is a function in the matplotlib.patches module
        legend_list.append(Patch(
            color = COLORS[cl],
            label = '{}: ({})'.format(CLASSES[cl], len(polygon_list[cl]))))

        for polygon in polygon_list[cl]:
            if scaler is not None:
                # affinity is a function from shapely
                polygon_rescale = affinity.scale(polygon, xfact = scaler[1],
                                                 yfact = scaler[0],
                                                 origin = [0., 0., 0.])
            else:
                polygon_rescale = polygon
            # PolygonPatch is a function from descartes.patch module
            # polygon_list is in relative coordinates and they are
            # generated from get_polygon_list and are further
            # converted to raster coordinates through scaler.
            patch = PolygonPatch(polygon = polygon_rescale, color = COLORS[cl],
                                 lw = 0, alpha = alpha, zorder = ZORDER[cl])
            ax.add_patch(patch)
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return legend_list


def plot_image(img, ax, image_id, image_key, selected_channel = None):
    '''
    Plot an selected channels of img into ax.
    :param img:
    :param ax:
    :param image_id:
    :param image_key:
    :param selected_channel:
    :return:
    '''
    title_suffix = ''
    if selected_channel is not None:
        img = img[:, :, selected_channel]
        title_suffix = '(' + ','.join(repr(i) for i in selected_channel) + ')'
    ax.imshow(img)
    ax.set_title(image_id + '-' + image_key + title_suffix)
    ax.set_xlabel(img.shape[0])
    ax.set_ylabel(img.shape[1])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_overlay(img, ax, image_id, image_key, polygon_list, scaler = [1., 1.],
                 x_range = None, y_range = None, label = None, alpha = 1.0,
                 rgb = False):
    '''
    Plot image with polygon overlays
    :param img:
    :param ax:
    :param image_id:
    :param image_key:
    :param polygon_list:
    :param scaler:
    :return:
    '''
    # cm is a function from matplotlib
    if not x_range:
        x_range = [0, img.shape[0]]
    if not y_range:
        y_range = [0, img.shape[1]]
    if rgb:
        ax.imshow(scale_percentile(img), vmax=1., vmin=0.)
    else:
        ax.imshow(scale_percentile(rgb2gray(img)),
                  cmap = cm.gray, vmax = 1., vmin = 0.)
    ax.set_xlabel(x_range[1] - x_range[0])
    ax.set_ylabel(y_range[1] - y_range[0])
    legend = plot_polygon(polygon_list, ax, scaler, alpha = alpha)
    ax.set_title(image_id + '-' + image_key + '-Overlay')
    return legend


def scale_percentile(img):
    '''
    Scale an image's 1 - 99 percentiles into 0 - 1 for display
    :param img:
    :return:
    '''
    orig_shape = img.shape
    if len(orig_shape) == 3:
        img = np.reshape(img,
                         [orig_shape[0] * orig_shape[1], orig_shape[2]]
                         ).astype(np.float32)
    elif len(orig_shape) == 2:
        img = np.reshape(img, [orig_shape[0] * orig_shape[1]]).astype(np.float32)
    mins = np.percentile(img, 1, axis = 0)
    maxs = np.percentile(img, 99, axis = 0) - mins

    img = (img - mins) / maxs

    img.clip(0., 1.)
    img = np.reshape(img, orig_shape)

    return img


def get_image_area(image_id):
    '''
    Calculate the area of an image
    :param image_id:
    :return:
    '''
    xmax = grid_sizes[grid_sizes.ImageId == image_id].Xmax.values[0]
    ymin = grid_sizes[grid_sizes.ImageId == image_id].Ymin.values[0]

    return abs(xmax * ymin)


def image_stat(image_id):
    '''
    Return the statistics ofd an image as a pd dataframe
    :param image_id:
    :return:
    '''
    counts, total_area, mean_area, std_area = {}, {}, {}, {}
    img_area = get_image_area(image_id)

    for cl in CLASSES:
        polygon_list = get_polygon_list(image_id, cl)
        counts[cl] = len(polygon_list)
        if len(polygon_list) > 0:
            total_area[cl] = np.sum([poly.area for poly in polygon_list])\
                             / img_area * 100.
            mean_area[cl] = np.mean([poly.area for poly in polygon_list])\
                            / img_area * 100.
            std_area[cl] = np.std([poly.area for poly in polygon_list])\
                           / img_area * 100.

    return pd.DataFrame({'Class': CLASSES, 'Counts': counts,
                         'TotalArea': total_area, 'MeanArea': mean_area,
                         'STDArea': std_area})


def collect_stats():
    '''
    Collect the area statistics for all images and concatenate them
    :return:
    '''
    stats = []
    total_no = len(all_train_names) - 1

    for image_no, image_id in enumerate(all_train_names):
        stat = image_stat(image_id)
        stat['ImageId'] = image_id
        stats.append(stat)
        sys.stdout.write('\rCollecting class stats [{}{}] {}%'.\
                         format('=' * image_no,
                                ' ' * (total_no - image_no),
                                100 * image_no / total_no))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return pd.concat(stats)


def calculate_class_weights():
    '''
    :return: class-wise true-label-area / false-label-area as a dictionary
    '''
    df = collect_stats()
    df = df.fillna(0)
    df = df.pivot(index = 'Class', columns = 'ImageId', values = 'TotalArea')
    df = df.sum(axis=1)
    df = df / (2500. - df)
    return df.to_dict()


def plot_stats(value, title):
    '''
    Plot 2D grid plot of statistics of MeanArea, Counts, TotalArea, STDArea.
    :param value:
    :param title:
    :return:
    '''
    stats = collect_stats()
    pvt = stats.pivot(index='Class', columns='ImageId', values = value)
    pvt.fillna(0., inplace = True)
    fig, ax = plt.subplots(figsize = (10, 4))
    im = ax.imshow(pvt, interpolation = 'nearest', cmap = plt.cm.plasma,
                   extent = [0 ,25, 10, 0])
    ax.set_xlabel('Image')
    ax.set_ylabel('Class Type')
    ax.set_xticks(np.arange(0.5, 25.4, 1))
    ax.set_yticks(np.arange(0.5, 10.4, 1))
    ax.set_xticklabels(np.arange(1, 26))
    ax.set_yticklabels(pvt.index)
    ax.set_title(title)

    fig.colorbar(im)


def plot_bar_stats():
    stats = collect_stats()
    pvt = stats.pivot(index = 'Class', columns = 'ImageId', values = 'TotalArea')
    perc_area = np.cumsum(pvt, axis = 0)
    class_r = {}
    sns.set_style('white')
    sns.set_context({'figure.figsize': (12, 8)})

    for cl in CLASSES: class_r[CLASSES[cl]] = cl

    for cl in np.arange(1, 11):
        class_name = perc_area.index[-cl]
        class_id = class_r[class_name]
        ax = sns.barplot(x = perc_area.columns, y = perc_area.loc[class_name],
                         color = COLORS[class_id], label = class_name)
    ax.legend(loc = 2)
    sns.despine(left = True)
    ax.set_xlabel('Image ID')
    ax.set_ylabel('Class Type')
    ax.set_xticklabels(perc_area.columns, rotation = -60)


def jaccard_index(mask_1, mask_2):
    '''
    Calculate jaccard index between two masks
    :param mask_1:
    :param mask_2:
    :return:
    '''
    assert len(mask_1.shape) == len(mask_2.shape) == 2
    assert 0 <= np.amax(mask_1) <=1
    assert 0 <= np.amax(mask_2) <=1

    intersection = np.sum(mask_1.astype(np.float32) * mask_2.astype(np.float32))
    union = np.sum(mask_1.astype(np.float32) + mask_2.astype(np.float32)) - \
            intersection

    if union == 0:
        return 1.

    return intersection / union


def polygon_jaccard(final_polygons, train_polygons):
    '''
    Calcualte the jaccard index of two polygons, based on data type of
    shapely.geometry.MultiPolygon
    :param final_polygons:
    :param train_polygons:
    :return:
    '''
    return final_polygons.intersection(train_polygons).area /\
    final_polygons.union(train_polygons).area
