from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from models.Unet.unet_various import MobileNetV2_Layers


class MobileNetV2_Ft_Linear(nn.Module):
    "returns intermidiate and final layers as features. Concatenates features at the same spatial resolution"

    def __init__(self):
        super(MobileNetV2_Layers, self).__init__()
        features = list(models.mobilenet_v2(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            results.append(x)

        combined_results = []
        out = torch.cat((results[0], results[1]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[2], results[3]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[4], results[5], results[6]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[7], results[8], results[9], results[10], results[11], results[12], results[13]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[14], results[15], results[16], results[17]), dim=1)
        combined_results.append(out)

        return(combined_results)


class bilinear_upsample(nn.Module):

    def __init__(self):
        super(bilinear_upsample, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class MobileNetV2_Ft(nn.Module):

    def __init__(self, args):
        self.num_classes = args['num_classes']
        self.num_features = 1539

        super(MobileNetV2_Ft, self).__init__()
        self.down_stack = MobileNetV2_Layers()

        if args['use_fixed_features']:
            self.freeze_weights()

    def freeze_weights(self):
        for param in self.down_stack.parameters():
            param.requires_grad = False

    def forward(self, im):
        skips = self.down_stack(im)

        y = skips[-1]
        skips = reversed(skips[:-1])

        for x in skips:
            y = F.interpolate(y, scale_factor=2, mode="bilinear")
            y = torch.cat((x, y), dim=1)

        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        y = torch.cat((im, y), dim=1)

        return y

class MobileNetV2_Ft_Linear(MobileNetV2_Ft):

    def __init__(self, args):
        super(MobileNetV2_Ft_Linear, self).__init__(args)

        self.conv_layer = nn.Conv2d(self.num_features, self.num_classes, 1, 1, 0)

        # for param in self.conv_layer.parameters():
        #     param.requires_grad = False

    def forward(self, im):

      ft = super(MobileNetV2_Ft_Linear, self).forward(im)

      out = self.conv_layer(ft)

      return out, ft



