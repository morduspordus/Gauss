from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import read_sizes

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        """
        super().__init__()

        self.mean = args['image_normalize_mean']
        self.std = args['image_normalize_std']

        self._base_dir = args['d_path']
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')

        if split == 'train':
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        elif split == 'val':
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        elif split == 'all':
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        else:
            raise NotImplementedError

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        if self.args['use_augmentation']:
            _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationAug')
        else:
            _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.im_names = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.im_names.append(line + ".png")

        assert (len(self.images) == len(self.categories))

        self.orig_im_sizes = read_sizes(self.images, self.im_names)  # store original image heights and widths
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def name(self):
        return self.args['dataset_name']


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                sample = self.transform_tr(sample)
                sample['name'] = self.im_names[index]
                sample['orig_size'] = self.orig_im_sizes[index]

                sample['label'][sample['label'] == 254] = 255

                return sample
            elif split == 'val' or split == 'all':
                sample = self.transform_val(sample)
                sample['name'] = self.im_names[index]
                sample['orig_size'] = self.orig_im_sizes[index]

                sample['label'][sample['label'] == 254] = 255

                return sample


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args['base_size'], crop_size=self.args['crop_size']),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args['crop_size']),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

