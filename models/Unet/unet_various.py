from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class MobileNetV2_Layers(nn.Module):
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


class pix2pix_upsample(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0):
        super(pix2pix_upsample, self).__init__()
        self.upsample = nn.Sequential(
                             nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.upsample(x)
        return x


class bilinear_upsample(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(bilinear_upsample, self).__init__()
        self.conv_block = nn.Sequential(
                             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv_block(x)
        return x


class Unet_Main(nn.Module):

    def __init__(self, args, upsample_param, with_bilinear=False, with_double_bilinear=False, all_skips=False):

        super(Unet_Main, self).__init__()

        output_channels = args['num_classes']
        final_activation = args['final_activation']
        num_final_features = args['num_final_features']
        post_processing = args['post_processing']

        self.post_processing = None
        self.num_classes = output_channels
        self.final_activation = final_activation
        self.num_final_features = num_final_features
        self.upsample_param = upsample_param[0]
        self.final_upsamle_layer_features = upsample_param[1]


        if with_bilinear:
            self.upstack1 = bilinear_upsample(self.upsample_param[0][0], self.upsample_param[0][1])
            self.upstack2 = bilinear_upsample(self.upsample_param[1][0], self.upsample_param[1][1])
            self.upstack3 = bilinear_upsample(self.upsample_param[2][0], self.upsample_param[2][1])
            self.upstack4 = bilinear_upsample(self.upsample_param[3][0], self.upsample_param[3][1])
            if all_skips:
                self.upstack5 = bilinear_upsample(self.upsample_param[4][0], self.upsample_param[4][1])
        else:
            self.upstack1 = pix2pix_upsample(self.upsample_param[0][0], self.upsample_param[0][1])
            self.upstack2 = pix2pix_upsample(self.upsample_param[1][0], self.upsample_param[1][1])
            self.upstack3 = pix2pix_upsample(self.upsample_param[2][0], self.upsample_param[2][1])
            self.upstack4 = pix2pix_upsample(self.upsample_param[3][0], self.upsample_param[3][1])
            if all_skips:
                self.upstack5 = pix2pix_upsample(self.upsample_param[4][0], self.upsample_param[4][1])


        if with_double_bilinear:
            upsample_layer = bilinear_upsample(self.final_upsamle_layer_features, self.num_final_features)
        else:
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(self.final_upsamle_layer_features, self.num_final_features, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(self.num_final_features),
                nn.ReLU(inplace=True)
            )

        final_layer = nn.Sequential(
            nn.Conv2d(self.num_final_features, output_channels, kernel_size=3, stride=1, padding=1),
        )

        if all_skips:
            self.up_stack = [self.upstack1, self.upstack2, self.upstack3, self.upstack4, self.upstack5]
            feature_list = [final_layer]
        else:
            self.up_stack = [self.upstack1, self.upstack2, self.upstack3, self.upstack4]
            feature_list = [upsample_layer, final_layer]


        self.features = nn.Sequential(*feature_list)

        if post_processing is not None:
            self.post_processing = post_processing(num_final_features)
        else:
            self.post_processing = None

    def freeze_weights(self):
        for param in self.down_stack.parameters():
            param.requires_grad = False


    def forward(self, x):
      skips = self.down_stack(x)

      x = skips[-1]
      skips = reversed(skips[:-1])

      for up, skip in zip(self.up_stack, skips):
          x = up(x)
          x = torch.cat((x, skip), dim=1)

      if self.post_processing is not None:
          x = self.post_processing(self.features, x)
      else:
          x = self.features(x)

      if self.final_activation == 'softmax':
          x = torch.softmax(x, dim=1)
      elif self.final_activation == 'L2':
          x = F.normalize(x, p=2, dim=1)
      elif self.final_activation == 'sigmoid':
          x = torch.sigmoid(x)
      return x


class Unet_MobileNetV2(Unet_Main):
    "Unet based on MobileNetV2 features"
    def __init__(self, args):

        upsample_param = ([(800, 512), (1056, 256), (352, 128), (176, 64)], 112)
        super(Unet_MobileNetV2, self).__init__(args, upsample_param)
        self.down_stack = MobileNetV2_Layers()

        if args['use_fixed_features']:
            self.freeze_weights()


class Unet_Resnet_18(Unet_Main):
    def __init__(self, args):
        upsample_param = ([(512, 512), (768, 256), (384, 128), (192, 64)], 128)
        #upsample_param = [(2048, 512), (1240, 256), (512, 128), (256, 64), 128]
        super(Unet_Resnet_18, self).__init__(args,upsample_param)

        self.main_model = smp.Unet('resnet18', encoder_weights='imagenet')
        #self.main_model = smp.Unet('xception', encoder_weights='imagenet')

        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()


class Unet_Resnet_50(Unet_Main):
    def __init__(self, args):
        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64)], 128]
        super(Unet_Resnet_50, self).__init__(args, upsample_param)

        self.main_model = smp.Unet('resnet50', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()


class Unet_se_resnet50(Unet_Main):
    def __init__(self, args):
        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64)], 128]
        super(Unet_se_resnet50, self).__init__(args, upsample_param)

        self.main_model = smp.Unet('se_resnet50', encoder_weights='imagenet', classes=output_channels)
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_se_resnext50_32x4d(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):
        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64)], 128]

        super(Unet_se_resnext50_32x4d, self).__init__(args, upsample_param)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_se_resnext50_32x4d_all_skips(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):

        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64), (128,  args['num_final_features']-3)], 128]
        super(Unet_se_resnext50_32x4d_all_skips, self).__init__(args, upsample_param, all_skips=True)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_se_resnext50_32x4d_bilinear(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):
        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64)], 128]
        super(Unet_se_resnext50_32x4d_bilinear, self).__init__(args, upsample_param, with_bilinear=True)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_se_resnext50_32x4d_bilinear_all_skips(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):
        upsample_param = [ [(2048, 512), (1536, 256), (768, 128), (384, 64), (128,  args['num_final_features']-3)], 128]
        super(Unet_se_resnext50_32x4d_bilinear_all_skips, self).__init__(args, upsample_param, with_bilinear=True, all_skips=True)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()




class Unet_se_resnext50_32x4d_double_bilinear(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):
        upsample_param = [[(2048, 512), (1536, 256), (768, 128), (384, 64)], 128]
        super(Unet_se_resnext50_32x4d_double_bilinear, self).__init__(args, upsample_param,
                                                                      with_bilinear=True,
                                                                      with_double_bilinear=True)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_MobileNetV2_double_bilinear(Unet_Main):
    "Unet based on MobileNetV2 features"
    def __init__(self, args):

        upsample_param = [[(800, 512), (1056, 256), (352, 128), (176, 64)], 112]
        super(Unet_MobileNetV2_double_bilinear, self).__init__(args, upsample_param,
                                                                      with_bilinear=True,
                                                                      with_double_bilinear=True)
        self.down_stack = MobileNetV2_Layers()

        if args['use_fixed_features']:
            self.freeze_weights()

class Unet_se_resnext50_32x4d_larger(Unet_Main):
    "Unet based on se_resnext50_32x4d features"

    def __init__(self, args):
        upsample_param = [[(2048, 1024), (2048, 512), (1024, 256), (512, 192)], 256]
        num_final_features = 256
        super(Unet_se_resnext50_32x4d_larger, self).__init__(args, upsample_param)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()


class Unet_Xception(Unet_Main):
    def __init__(self, args):
        upsample_param = [[(2048, 512), (1240, 256), (512, 128), (256, 64)], 128]
        super(Unet_Xception, self).__init__(args, upsample_param)

        self.main_model = smp.Unet('xception', encoder_weights='imagenet')

        self.down_stack = self.main_model.encoder

        if args['use_fixed_features']:
            self.freeze_weights()


class SemSegModel(nn.Module):
    def __init__(self, args):
        super(SemSegModel, self).__init__()
        self.use_fixed_features = args['use_fixed_features']
        self.output_channels = args['num_classes']


    def fix_weights(self):
        if self.use_fixed_features:
            for param in self.main_model.encoder.parameters():
                param.requires_grad = False


    def forward(self,  x):
        x = self.main_model(x)

        return x


class Unet_se_resnext50_32x4d_original(SemSegModel):
    def __init__(self, args):
        super(Unet_se_resnext50_32x4d_original, self).__init__(args)
        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet', classes=self.output_channels, activation='softmax')
        self.fix_weights()

