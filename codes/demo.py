import argparse
import matplotlib.pyplot as plt
import os.path
import glob
import cv2
import numpy as np
import torch

from data.util import imresize, modcrop
import utils.util as util
import models.modules.sft_arch as sft
from data.dstl_dataset.test_dataset import DstlDataset
import options.options as option


# model_path = '../experiments/pretrained_models/sft_net_torch.pth' # torch version
model_path = '/workspace/BasicSR/experiments/SFTGANx4_dstl/models/258000_G.pth'

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=True)
opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

train_set = DstlDataset(opt['datasets']['train'])

if 'torch' in model_path:  # torch version
    model = sft.SFT_Net_torch()
else:
    model = sft.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()

print('sftgan testing...')

for i in range(25):
    data = train_set[i]
    img_HR = data['HR']
    img_LR = data['LR']
    seg = data['seg']
    output = model((img_LR.cuda(), seg.cuda())).data
    
    util.save_img(util.tensor2img(img_HR.squeeze()),
        os.path.join('../results', '{}_hr.png'.format(i)))

    util.save_img(util.tensor2img(output.squeeze()),
        os.path.join('../results', '{}_fake.png'.format(i)))

