import torch
import numpy as np
import os
from dataloaders.utils import img_denormalize
from PIL import Image


class ImageSaver(object):
    def __init__(self, args, images_dir, out_multiplier=1, decode_segmap=None, add_subdir=False, with_gt=False,
                 with_orig_im=False, with_orig_size=False, divider=10):
        # if decode_segmap is None, saves grayscale output, otherwise creates color output
        # out_multiplier is multiplied with the output labels
        # out_sizes is a dictionary, with key=filename, of the output size to use for saving image.

        self.with_gt = with_gt
        self.divider_width = divider
        self.with_orig_im = with_orig_im
        self.with_orig_size=with_orig_size
        self.add_subdir = add_subdir
        self.images_dir = images_dir
        self.decode_segmap = decode_segmap
        self.out_multiplier = out_multiplier
        self.mean = args['image_normalize_mean']
        self.std = args['image_normalize_std']

    def save(self, pred, labels, inputs, file_names, orig_size):

        if type(pred) is tuple:  # in case model outputs a tuple, first element is prediction
            pred = pred[0]

        im_size = pred.shape[2]

        if len(list(pred.size())) == 4:  # convert from prob to labels
            _, pred = torch.max(pred, dim=1)

        divider = np.zeros((im_size, self.divider_width, 3)) + 255

        pred = pred.detach().cpu()
        labels = labels.detach().cpu()

        for jj in range(inputs.size()[0]):

            file_name = file_names[jj]

            file_name = file_name.replace('.jpg', '.png')

            if self.add_subdir:
                save_name = os.path.join(self.images_dir, file_name)
            else:
                save_name = os.path.join(self.images_dir, file_name)

            out_im = self.out_multiplier * pred[jj]
            out_im = np.array(out_im).astype(np.uint8)

            if self.decode_segmap is not None:
                out_im = 255*self.decode_segmap(out_im)

            out_im = np.int8(out_im)

            if self.with_gt:
                assert(self.decode_segmap is not None)

                gt_val_labels = self.out_multiplier * labels[jj]
                gt_val_labels = np.array(gt_val_labels).astype(np.uint8)

                label_map_gt_val = self.decode_segmap(gt_val_labels)
                label_map_gt_val = np.int8(255 * label_map_gt_val)
                label_map_gt_val = np.concatenate((label_map_gt_val, divider), axis=1)
                out_im = np.concatenate((label_map_gt_val, out_im), axis=1)

            if self.with_orig_im:
                img = inputs[jj].cpu().numpy()
                img = img_denormalize(img, self.mean, self.std)
                img = np.concatenate((img, divider), axis=1)
                out_im = np.concatenate((img, out_im), axis=1)

            if self.decode_segmap is None:
                img = Image.fromarray(np.int8(out_im), mode='L')
                if self.with_orig_size:
                    height = orig_size['height'][jj]
                    width = orig_size['width'][jj]
                    img = img.resize((height, width),  Image.NEAREST)

            else:
                img = Image.fromarray(np.int8(out_im), mode='RGB')

            img.save(save_name)

