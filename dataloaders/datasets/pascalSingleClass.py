from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import read_sizes


class PascalSingleClass(Dataset):
    """
    pascalSingleClass
        if  args['single_class'] = -1, extracts all classes
        if args['single_class'] > 0, extracts class given by args['single_class']
        args['d_path']: dataset path
        args['annotaton_dir']: if None, no ground truth is provided

    """
    __name__ = 'PascalSingleClass'
    NUM_CLASSES = 21
    def __init__(self,
                 args,
                 split='train'
                 ):
        """
        :param split: train
        """
        super().__init__()

        self.transform_to_PIL = transforms.ToPILImage()

        self.args = args

        self._base_dir = args['d_path']
        self._image_sets_dir = os.path.join(self._base_dir, args['image_sets_dir'])
        self._images_dir = os.path.join(self._base_dir, args['images_dir'])
        self.annotation_dir = args['annotation_dir']  # target labels are in this subdirectory

        self.split = split
        self.single_class = args['single_class']

        self.mean = args['image_normalize_mean']
        self.std = args['image_normalize_std']

        self.negative_class = args['negative_class']  # uses other classes in the dataset as negative classes
        self.join_class_dir = args['join_class_dir']  # files are located in 'class' subdirectories

        self.img_paths = []
        self.mask_paths = []
        self.img_names = []


        if self.single_class > 0:
            if split == 'train':
                image_names_files = os.path.join(self._image_sets_dir, 'train', str(self.single_class)+'.txt')
            elif split == 'val':
                image_names_files = os.path.join(self._image_sets_dir, 'val', str(self.single_class) +'.txt')
            elif split == 'all':
                image_names_files = os.path.join(self._image_sets_dir, 'all', str(self.single_class)+'.txt')
            else:
                print('Split option {} is not available.'.format(split))
                raise NotImplementedError

            self.img_names = [line.rstrip('\n') for line in open(image_names_files)]

            image_files_path = self._images_dir
            if self.join_class_dir:
                image_files_path = os.path.join(self._images_dir, str(self.single_class))

            self.img_paths = [image_files_path + '\\' + name for name in self.img_names]

            if self.annotation_dir is None: # no ground truth
                self.mask_paths = [None] * len(self.img_paths)
            else:
                if self.join_class_dir:
                    annotation_dir = os.path.join(self._base_dir, self.annotation_dir, str(self.single_class))
                else:
                    annotation_dir = os.path.join(self._base_dir, self.annotation_dir)
                self.mask_paths = [annotation_dir + '\\' + name.replace('.jpg', '.png') for name in self.img_names]

            if self.negative_class and split == 'train': # use negative samples only for training
                self.handle_negative_class()
        else:
            self.true_class = []

            for i in range(1, self.NUM_CLASSES):
                if split == 'train':
                    image_names_files = os.path.join(self._image_sets_dir, 'train', str(i) + '.txt')
                elif split == 'val':
                    image_names_files = os.path.join(self._image_sets_dir, 'val', str(i) + '.txt')
                elif split == 'all':
                    image_names_files = os.path.join(self._image_sets_dir, 'all', str(i) + '.txt')
                else:
                    print('Split option {} is not available.'.format(split))
                    raise NotImplementedError

                img_names = [line.rstrip('\n') for line in open(image_names_files)]

                image_files_path = self._images_dir
                if self.join_class_dir:
                    image_files_path = os.path.join(self._base_dir, str(i))

                img_paths = [image_files_path + '\\' + name for name in img_names]

                self.img_names.extend(img_names)
                self.img_paths.extend(img_paths)
                self.true_class.extend([i] * len(img_paths))

                if self.annotation_dir is None:
                    self.mask_paths.extend([None] * len(img_paths))
                else:
                    if self.join_class_dir:
                        annotation_dir_extended = os.path.join(self.annotation_dir, str(i))
                    else:
                        annotation_dir_extended = self.annotation_dir
                    annotation_dir_extended = os.path.join(self._base_dir, annotation_dir_extended)
                    mask_paths = [annotation_dir_extended + '\\' + name.replace('.jpg', '.png') for name in img_names]
                    self.mask_paths.extend(mask_paths)


        self.orig_im_sizes = read_sizes(self.img_paths, self.img_names) # store original image heights and widths

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_paths)))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if _target == None:
            h,w = _img.size
            _target = np.zeros([w, h], dtype=int)
            _target = self.transform_to_PIL(_target)

        img_name = self.img_names[index]
        if self.single_class > 0:
            true_class = 1
        else:
            true_class = self.true_class[index]
        if self.negative_class:
            true_class = self.true_class[index]

        sample = {'image': _img, 'label': _target}


        if self.split == "train":
            sample = self.transform_tr(sample)
            sample['name'] = self.img_names[index]
        elif self.split == 'val':
            sample = self.transform_val(sample)
            sample['name'] = self.img_names[index]
        elif self.split == 'all':
            sample = self.transform_val(sample)
            sample['name'] = self.img_names[index]
        else:
            print('split {} not available.'.format(self.split))
            raise NotImplementedError

        if self.single_class > 0:
            sample['label'] [ sample['label'] == self.single_class ] = 1

        sample['image_class'] = true_class
        sample['name'] = img_name
        sample['orig_size'] = self.orig_im_sizes[index]

        return sample


    def __len__(self):
        return len(self.img_paths)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.img_paths[index]).convert('RGB')

        if self.mask_paths[index] != None:
            _target = Image.open(self.mask_paths[index])
        else:
            _target = None

        return _img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(size=self.args['crop_size']),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args['crop_size']),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'PascalSingleClass:'+self.args['dataset_name']+'(split=' + str(self.split) + ')'

    def name(self):
        return self.args['dataset_name']


    def handle_negative_class(self):

            neg_img_names = []
            neg_img_paths = []

            for i in range(1, self.NUM_CLASSES ):
                if i != self.single_class:
                    if self.split == 'train':
                        image_names_files = os.path.join(self._image_sets_dir, 'train', str(i)+'.txt')
                    elif self.split == 'val':
                        image_names_files = os.path.join(self._image_sets_dir, 'val', str(i)+'.txt')
                    elif self.split == 'all':
                        image_names_files = os.path.join(self._image_sets_dir, 'all', str(i)+'.txt')
                    else:
                        print('Split option {} is not available.'.format(self.split))
                        raise NotImplementedError

                    image_files_path = self._images_dir
                    if self.join_class_dir:
                        image_files_path = os.path.join(self._images_dir, str(i))

                    img_names = [line.rstrip('\n') for line in open(image_names_files)]
                    img_paths = [image_files_path + '\\' + name for name in img_names]

                    neg_img_names.extend(img_names)
                    neg_img_paths.extend(img_paths)


            self.true_class = [1] * len(self.img_names)

            num_positive = len(self.img_names)
            num_negative = len(neg_img_names)

            ratio = num_negative//num_positive

            temp_true_class = self.true_class*1
            temp_img_names = self.img_names*1
            temp_mask_paths = self.mask_paths*1
            temp_img_paths = self.img_paths*1

            if ratio > 1:
                for i in range(ratio-1):
                    self.true_class.extend(temp_true_class)
                    self.img_names.extend(temp_img_names)
                    self.mask_paths.extend(temp_mask_paths)
                    self.img_paths.extend(temp_img_paths)

            neg_class = [0] * len(neg_img_names)
            neg_mask_paths = [None] * len(neg_img_names)

            self.true_class.extend(neg_class)
            self.img_names.extend(neg_img_names)
            self.img_paths.extend(neg_img_paths)
            self.mask_paths.extend(neg_mask_paths)
