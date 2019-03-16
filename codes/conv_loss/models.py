import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

import pretrainedmodels.models.pnasnet as pnasnet


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


def pnasnet_features(self, x):
    x_conv_0 = self.conv_0(x)
    x_stem_0 = self.cell_stem_0(x_conv_0)
    x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
    x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
    x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
    x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
    x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
    #x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
    #x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
    #x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
    #x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
    #x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
    #x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
    #x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
    #x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
    return x_cell_3

pnasnet.PNASNet5Large.features = pnasnet_features

class PNasNetFeatureExtractor(nn.Module):
    def __init__(self,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(PNasNetFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.model = pnasnet.PNASNet5Large(num_classes=1001)
        self.model.load_state_dict(model_zoo.load_url(
            'http://data.lip6.fr/cadene/pretrainedmodels/pnasnet5large-bf079911.pth'))
        if self.use_input_norm:
            mean = torch.Tensor([0.5-1, 0.5-1, 0.5-1]).view(1, 3, 1, 1).to(device)
            # [0.5-1, 0.5-1, 0.5-1] if input in range [-1,1]
            std = torch.Tensor([0.5*2, 0.5*2, 0.5*2]).view(1, 3, 1, 1).to(device)
            # [0.5*2, 0.5*2, 0.5*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = self.model.features
        # No need to BP to variable
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.avg_pool = nn.AvgPool2d(12, stride=12, padding=0)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        output = self.avg_pool(output)
        return output
