import argparse
import matplotlib.pyplot as plt
import os.path
import glob
import cv2
import numpy as np
import torch

from data.util import imresize, modcrop
import utils.util as util
from models import create_model
from data.dstl_dataset.dataset import denormalize
from data.dstl_dataset.dataset_test import DstlDataset
import options.options as option


#model_path = '/workspace/BasicSR/experiments/RRDB_PSNR_x4_DSTL/models/100000_G.pth'
model_path = '/workspace/BasicSR/experiments/RRDB_ESRGAN_x4_DSTL/models/35000_G.pth'

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=True)
opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

dataset = DstlDataset(opt['datasets']['val'])

model = create_model(opt).netG.module
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('sftgan testing...')

for i in range(len(dataset)):
    data = dataset[i]
    img_HR = data['HR']
    img_LR = data['LR']
    output = model(img_LR.cuda()).data

    img_HR = img_HR.numpy()
    output = output.squeeze().cpu().numpy()

    img_HR = denormalize(img_HR)
    output = denormalize(output)

    psnr = util.calculate_psnr(img_HR, output)
    ssim = util.calculate_ssim(img_HR, output)
    print(psnr, ssim)

    util.save_img(img_HR, os.path.join('../results', '{}_hr.png'.format(i)))
    util.save_img(output, os.path.join('../results', '{}_fake.png'.format(i)))
