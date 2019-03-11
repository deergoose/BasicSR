import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import os

_RESULT_DIR = '/Users/zhoufang/Desktop/lu/esrgan/results/gan_paper/results_vgg_6_100k/results_vgg_34'


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=16, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF


pixel_mean = [432.27961319, 468.29362853, 335.24936232]
pixel_max = 2000.0 #2047


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
    im = im / 2100 * 255
    return im.astype(np.uint8)


def read_im(im_dir, im_name):
    im = cv2.imread(os.path.join(im_dir, im_name))
    #im = im[:128, :128, :]
    im = normalize(im.astype(np.float) / 255.0 * 2100)
    im = im[np.newaxis, ...]
    return torch.from_numpy(np.ascontiguousarray(
        np.transpose(im, (0, 3, 1, 2)))).float()


def main():
    opt = {
        'gpu_ids': None
    }
    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    cri_fea = nn.L1Loss().to(device)
    #cri_fea = nn.MSELoss().to(device)

    netF = define_F(opt, use_bn=False).to(device)

    im_hr = read_im(_RESULT_DIR, '15_hr.png')
    im_fake = read_im(_RESULT_DIR, '15_fake.png')

    real_fea = netF(im_hr)
    fake_fea = netF(im_fake)
    l_g_fea = cri_fea(fake_fea, real_fea)
    print l_g_fea

    real_feat = np.transpose(real_fea.numpy()[0], (1, 2, 0))
    fake_feat = np.transpose(fake_fea.numpy()[0], (1, 2, 0))
    feat_diff = np.absolute(real_feat - fake_feat)
    print 'sum: ', np.sum(feat_diff)

    for i in range(real_feat.shape[2]):
        im_real_feat = cv2.normalize(
            real_feat[..., i], None, 0, 255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        im_fake_feat = cv2.normalize(
            fake_feat[..., i], None, 0, 255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        im_feat_diff = cv2.normalize(
            feat_diff[..., i], None, 0, 255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        print np.sum(feat_diff[..., i])
        cv2.imshow('real', im_real_feat)
        cv2.imshow('fake', im_fake_feat)
        cv2.imshow('diff', im_feat_diff)
        cv2.waitKey()


if __name__ == '__main__':
    main()
