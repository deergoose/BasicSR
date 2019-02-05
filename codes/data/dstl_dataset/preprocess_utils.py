import cv2
import numpy as np


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    diff_nx = (data.shape[1] - shape[1])
    diff_ny = (data.shape[2] - shape[2])

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[:, offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[1] == shape[1]
    assert cropped.shape[2] == shape[2]
    return cropped


def adjust_size(img, scale):
    assert len(img.shape) == 3
    w, h = img.shape[0], img.shape[1]
    img = img[:w-w%scale, :h-h%scale, :]
    return img


# img has shape [h, w, c].
def blur(img, scale):
    assert len(img.shape) == 3

    x = cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    #result = []
    #for i in range(img.shape[2]):
    #    x = scipy.ndimage.interpolation.zoom(img[..., i], (1./r),
    #        prefilter=False)
    #    result.append(scipy.ndimage.interpolation.zoom(x, r,
    #        prefilter=False))
    #img = np.stack(result, axis=2)

    return img


def downsample(img, scale):
    assert len(img.shape) == 3
    return cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_CUBIC)


# img = [h, w, c], label = [h, w, c]
# rotation is the grain of angles.
def rand_rotate_and_crop(img, label, crop_size, rotation=8, reflection=True,
        verbose=False):
    x_img, y_img = img.shape[0], img.shape[1]

    angle = 360. * np.random.randint(0, rotation) / rotation
    radian = 2. * np.pi * angle / 360.
    if verbose:
        print('Rotation angle : {0}(degree), {1: 0.2f}(radian)'. \
            format(int(angle), radian))

    crop_size_new = int(np.ceil(float(crop_size) * (abs(np.sin(radian)) +
                                abs(np.cos(radian)))))
    rot_mat = cv2.getRotationMatrix2D((float(crop_size_new) / 2.,
                                    float(crop_size_new) / 2.), angle, 1.)
    crop_diff = int((crop_size_new - crop_size) / 2.)

    x_start = x_img - crop_size_new
    y_start = y_img - crop_size_new

    if x_start <= 0 or y_start <= 0:
        return img, label

    x_base = np.random.randint(0, x_start)
    y_base = np.random.randint(0, y_start)
    if verbose:
        print('x_base {} for No. {} image'.format(x_base, id))
        print('y_base {} for No. {} image'.format(y_base, id))

    img_crop = img[x_base:x_base+crop_size_new, y_base:y_base+crop_size_new, :]
    label_crop = label[x_base:x_base+crop_size_new, y_base:y_base+crop_size_new, :]

    img_rot = cv2.warpAffine(img_crop, rot_mat,
                            (crop_size_new, crop_size_new))
    label_rot = cv2.warpAffine(label_crop, rot_mat,
                            (crop_size_new, crop_size_new))

    x_step = 1 if not reflection else [-1, 1][np.random.randint(0, 2)]
    y_step = 1 if not reflection else [-1, 1][np.random.randint(0, 2)]

    img = img_rot[crop_diff:crop_diff+crop_size:, crop_diff:crop_diff+crop_size, :] \
                [::x_step, ::y_step, :]
    label = label_rot[crop_diff:crop_diff+crop_size, crop_diff:crop_diff+crop_size, :] \
                [::x_step, ::y_step, :]
    return img, label
