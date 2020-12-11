from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from models.Unet.unet_various import MobileNetV2_Layers


class MobileNetV2_Ft_Linear(nn.Module):
    "returns intermediate and final layers as features. Concatenates features at the same spatial resolution"

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


# class MobileNetV2_Ft_LinearFixed(MobileNetV2_Ft):
#
#     def __init__(self, args):
#         super(MobileNetV2_Ft_LinearFixed, self).__init__(args)
#
#         self.W_aug_matrix = args['W_aug_matrix']
#         self.W_aug_matrix = torch.transpose(self.W_aug_matrix, 0, 1)
#
#         self.device = args['device']
#
#     def forward(self, im):
#
#       ft_ = super(MobileNetV2_Ft_LinearFixed, self).forward(im)
#
#       [n, d, h, w] = list(ft_.size())
#       ft = torch.cat([ft_, torch.ones(n, 1, h, w).to(self.device)], 1)
#
#       ft = torch.transpose(ft, 0, 1)
#       ft = torch.flatten(ft, start_dim=1)
#       ft = torch.transpose(ft, 0, 1)
#
#       res = torch.matmul(ft, self.W_aug_matrix)
#       res = res.view(n, h, w, self.num_classes)
#       res = torch.transpose(res, 1, 3)
#       res = torch.transpose(res, 2, 3)
#
#       return res, ft_
#

class MobileNetV2_Ft_LinearFixed(MobileNetV2_Ft):

    def __init__(self, args):
        super(MobileNetV2_Ft_LinearFixed, self).__init__(args)

        self.device = args['device']
        requires_grad = args['mean_requires_grad']

        self.mean = torch.nn.Parameter(args['mean'], requires_grad=requires_grad)
        self.var = torch.nn.Parameter(args['var'], requires_grad=requires_grad)

        self.num_classes, d = list(self.mean.size())

    def forward(self, im):

        ft = super(MobileNetV2_Ft_LinearFixed, self).forward(im)
        [n, d, h, w] = list(ft.size())

        ft = torch.transpose(ft, 0, 1)
        ft = torch.flatten(ft, start_dim=1)
        ft = torch.transpose(ft, 0, 1)


        for cl in range(self.num_classes):
            mean_cl = self.mean[cl, :]
            var_cl = self.var[cl, :]

            loss_cl = (ft - mean_cl) ** 2
            loss_cl = loss_cl / (2 * var_cl)
            loss_cl = torch.sum(loss_cl, dim=1)

            logvar = (1 / 2) * torch.log(var_cl)

            logsigmas = torch.sum(logvar)

            loss_cl = loss_cl + logsigmas

            if cl == 0:
                res = loss_cl[:,None]
            else:
                res = torch.cat((res, loss_cl[:,None]), dim=1)

        res = res.view(n, h, w, self.num_classes)
        res = torch.transpose(res, 1, 3)
        res = torch.transpose(res, 2, 3)

        return -res, ft, self.mean, self.var

        # ft = super(MobileNetV2_Ft_LinearFixed, self).forward(im)
        # [n, d, h, w] = list(ft.size())
        #
        # ft = torch.transpose(ft, 0, 1)
        # ft = torch.flatten(ft, start_dim=1)
        # ft = torch.transpose(ft, 0, 1)
        #
        # ft_ = ft.unsqueeze(dim=2)
        # ft_ = ft_.repeat(1, 1, 3)
        #
        # mean_trans = torch.transpose(self.mean, 0, 1)
        # res = ft_ - mean_trans
        #
        # res = res ** 2
        # res = -res
        # var_trans = torch.transpose(self.var, 0, 1)
        # res = res * (1/(0.5*var_trans))
        # res = torch.sum(res, dim=1)
        #
        # logvar = -(1 / 2) * torch.log(self.var)
        # logsigmas = torch.sum(logvar, dim=1)
        #
        # res = res + logsigmas
        #
        # res = res.view(n, h, w, self.num_classes)
        # res = torch.transpose(res, 1, 3)
        # res = torch.transpose(res, 2, 3)
        #
        # return res, ft, self.mean, self.var

        #
        #
        # W_aug_matrix = self.mean / self.var
        # mean_by_var = -self.mean / (2 * self.var)
        # mean_t = torch.transpose(self.mean, 0, 1)
        #
        # bias = torch.matmul(mean_by_var, mean_t)
        # diag = torch.diagonal(bias)
        # logvar = -(1 / 2) * torch.log(self.var)
        # logsigmas = torch.sum(logvar, dim=1)
        # diag = diag + logsigmas
        #
        # # if self.print_mean:
        # #     print("mean", self.mean[0,1], self.mean[0,10], self.mean[1,1000], self.mean[2,200])
        # #     print("var", self.var[0,1], self.var[0,10], self.var[1,1000], self.var[2,200])
        #
        # # if self.class_pr is not None:
        # #     nll = -torch.log(self.class_pr)
        # #     diag = diag + nll
        #
        # diag = torch.unsqueeze(diag, dim=1)
        # W_aug_matrix = torch.cat((W_aug_matrix, diag), dim=1)
        # W_aug_matrix = torch.transpose(W_aug_matrix, 0, 1)
        #
        # ft = super(MobileNetV2_Ft_LinearFixed, self).forward(im)
        #
        # # ft_sq = ft ** 2
        # # ft_sq = torch.transpose(ft_sq, 0, 1)
        # # ft_sq = torch.flatten(ft_sq, start_dim=1)
        # # ft_sq = torch.transpose(ft_sq, 0, 1)
        # # ft_sq = ft_sq / (2 * self.var)
        #
        # [n, d, h, w] = list(ft.size())
        # ft = torch.cat([ft, torch.ones(n, 1, h, w).to(self.device)], 1)
        #
        # ft = torch.transpose(ft, 0, 1)
        # ft = torch.flatten(ft, start_dim=1)
        # ft = torch.transpose(ft, 0, 1)
        #
        # res = torch.matmul(ft, W_aug_matrix)
        #
        #
        # ft_sq = torch.sum(ft_sq, dim=1)
        # res += ft_sq
        #
        #
        # res = res.view(n, h, w, self.num_classes)
        # res = torch.transpose(res, 1, 3)
        # res = torch.transpose(res, 2, 3)
        #
        # return res, ft[:, 0:self.num_features], self.mean, self.var


#
#
# class MobileNetV2_Ft_LinearFixed(MobileNetV2_Ft):
#
#     def __init__(self, args):
#         super(MobileNetV2_Ft_LinearFixed, self).__init__(args)
#
#         self.W_aug_matrix = args['W_aug_matrix']
#         self.device = args['device']
#         self.num_features = args['num_features']
#
#         self.conv_layer = nn.Conv2d(self.num_features, self.num_classes, 1, 1, 0)
#
#         # for param in self.conv_layer.parameters():
#         #     param.requires_grad = False
#
#         to_replace = self.W_aug_matrix[:, self.num_features:].flatten()
#         self.conv_layer.bias[:] = to_replace
#
#
#         to_replace = self.W_aug_matrix[:, 0:self.num_features]
#         to_replace = to_replace.unsqueeze(dim=2)
#         to_replace = to_replace.unsqueeze(dim=2)
#
#         self.conv_layer.weight.data = to_replace
#
#
#     def forward(self, im):
#
#       ft = super(MobileNetV2_Ft_LinearFixed, self).forward(im)
#
#       out = self.conv_layer(ft)
#
#
#       return out, ft
#
#
#
#
#
#
