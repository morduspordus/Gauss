import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # m = np.min(img)
        # M = np.max(img)
        # img = (img-m)/(M-m+0.000000001)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample['image'] = img
        sample['label'] = mask

        return sample

class NormalizeImage(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        sample['image'] = img
        sample['label'] = mask

        if 'colorquant' in sample:
            colorquant = sample['colorquant']
            colorquant = np.array(colorquant).astype(np.float32).transpose((0, 1))
            sample['colorquant'] = colorquant

        if 'edges_h' in sample:
            edges_v = sample['edges_v']
            edges_v = np.array(edges_v).astype(np.float32).transpose((0, 1))
            sample['edges_v'] = edges_v

            edges_h = sample['edges_h']
            edges_h = np.array(edges_h).astype(np.float32).transpose((0, 1))
            sample['edges_h'] = edges_h

        if 'dist_h' in sample:
            dist_v = sample['dist_v']
            dist_v = np.array(dist_v).astype(np.float32).transpose((0, 1))
            sample['dist_v'] = dist_v

            dist_h = sample['dist_h']
            dist_h = np.array(dist_h).astype(np.float32).transpose((0, 1))
            sample['dist_h'] = dist_h
        return sample

class ToTensorImage(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            to_return = {'image': img, 'label': mask}

            if 'colorquant' in sample:
                colorquant = sample['colorquant']
                colorquant = colorquant.transpose(Image.FLIP_LEFT_RIGHT)
                to_return['colorquant'] = colorquant

            if 'edges_h' in sample:
                edges_h = sample['edges_h']
                edges_h = edges_h.transpose(Image.FLIP_LEFT_RIGHT)
                to_return['edges_h'] = edges_h

                edges_v = sample['edges_v']
                edges_v = edges_v.transpose(Image.FLIP_LEFT_RIGHT)
                to_return['edges_v'] = edges_v

            if 'dist_h' in sample:
                dist_h = sample['dist_h']
                dist_h = dist_h.transpose(Image.FLIP_LEFT_RIGHT)
                to_return['dist_h'] = dist_h

                dist_v = sample['dist_v']
                dist_v = dist_v.transpose(Image.FLIP_LEFT_RIGHT)
                to_return['dist_v'] = dist_v

        else:
            to_return = sample

        return to_return


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        to_return = {'image': img, 'label': mask}

        if 'colorquant' in sample:
            colorquant = sample['colorquant']
            colorquant = colorquant.resize(self.size, Image.NEAREST)
            to_return['colorquant'] = colorquant

        if 'edges_h' in sample:
            edges_h = sample['edges_h']
            edges_h = edges_h.resize(self.size, Image.NEAREST)
            to_return['edges_h'] = edges_h

            edges_v = sample['edges_v']
            edges_v = edges_v.resize(self.size, Image.NEAREST)
            to_return['edges_v'] = edges_v

        if 'dist_h' in sample:
            dist_h = sample['dist_h']
            dist_h = dist_h.resize(self.size, Image.NEAREST)
            to_return['dist_h'] = dist_h

            dist_v = sample['dist_v']
            dist_v = dist_v.resize(self.size, Image.NEAREST)
            to_return['dist_v'] = dist_v

        return to_return

def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
