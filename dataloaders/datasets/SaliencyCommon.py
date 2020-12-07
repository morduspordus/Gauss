from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import read_sizes

class SaliencyCommon(Dataset):
    """
    For training on Saliency datasets that have common structure, such as MSRA, ECSSD, DUT-OMRON
    """

    NUM_CLASSES = 2
    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param split: train/val

        """
        super().__init__()

        self._base_dir = args['d_path']
        self.split = [split]
        self.args = args
        self.mean = args['image_normalize_mean']
        self.std = args['image_normalize_std']
        self.read_edges = args['read_edges']
        self.read_dist = args['read_dist']
        self.read_colorquant = args['read_colorquant']

        if 'add_file_extension' in args:
            self.add_file_extension = args['add_file_extension']
        else:
            self.add_file_extension = False

        if split == 'train':
            image_names_files = os.path.join(self._base_dir, 'annotations\\train.txt')
        elif split == 'val':
            image_names_files = os.path.join(self._base_dir, 'annotations\\test.txt')
        elif split == 'all':
            image_names_files = os.path.join(self._base_dir, 'annotations\\all.txt')
        elif split == 'custom':
            image_names_files = os.path.join(self._base_dir, 'annotations', args['custom_split'])
        else:
            print('Split option {} is not available.'.format(split))
            raise NotImplementedError

        image_files_path = os.path.join(self._base_dir, 'images')
        edges_files_path = os.path.join(self._base_dir, 'canny')
        colorquant_files_path = os.path.join(self._base_dir, 'colorquant16')
        mask_files_path = os.path.join(self._base_dir, 'annotations\\labels')

        self.img_names = [line.rstrip('\n') for line in open(image_names_files)]

        if self.add_file_extension:
            self.img_names = [line + '.jpg' for line in self.img_names]

        self.img_paths = [image_files_path + '\\' + name for name in self.img_names]
        self.mask_paths = [mask_files_path + '\\' + name.replace('.jpg','.png') for name in self.img_names]
        if self.read_edges:
            # self.edge_paths_h = [edges_files_path + '\\' + name.replace('.jpg', '_h.png') for name in self.img_names]
            # self.edge_paths_v = [edges_files_path + '\\' + name.replace('.jpg', '_v.png') for name in self.img_names]
            self.edge_paths_h = [edges_files_path + '\\' + name.replace('.jpg', '.png') for name in self.img_names]
            self.edge_paths_v = [edges_files_path + '\\' + name.replace('.jpg', '.png') for name in self.img_names]
        if self.read_dist:
            self.dist_paths_h = [edges_files_path + '\\' + name.replace('.jpg', '_h_dist.png') for name in self.img_names]
            self.dist_paths_v = [edges_files_path + '\\' + name.replace('.jpg', '_v_dist.png') for name in self.img_names]
        if self.read_colorquant:
            self.colorquant_path = [colorquant_files_path + '\\' + name.replace('.jpg', '.png') for name in
                                     self.img_names]

        self.orig_im_sizes = read_sizes(self.img_paths, self.img_names) # store original image heights and widths

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return('Saliency')

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        img_name = self.img_names[index]

        sample = {'image': _img, 'label': _target}

        if self.read_edges:
            edges_h = Image.open(self.edge_paths_h[index]).convert('L')
            sample['edges_h'] = edges_h
            edges_v = Image.open(self.edge_paths_v[index]).convert('L')
            sample['edges_v'] = edges_v
        if self.read_dist:
            dist_h = Image.open(self.dist_paths_h[index]).convert('L')
            sample['dist_h'] = dist_h
            dist_v = Image.open(self.dist_paths_v[index]).convert('L')
            sample['dist_v'] = dist_v
        if self.read_colorquant:
            colorquant = Image.open(self.colorquant_path[index]).convert('L')
            sample['colorquant'] = colorquant

        for split in self.split:
            if split == "train" or split == 'trainval':
                sample = self.transform_tr(sample)
                sample['name'] = self.img_names[index]
            elif split == 'val' or split == 'test':
                sample['name'] = self.img_names[index]
                sample = self.transform_val(sample)
                sample['name'] = self.img_names[index]
            elif split == 'all':
                sample = self.transform_val(sample)
                sample['name'] = self.img_names[index]
            elif split == 'custom':
                sample = self.transform_custom(sample)
                sample['name'] = self.img_names[index]
            else:
                print('split {} not available.'.format(split))
                raise NotImplementedError

        if 'edges_h' in sample:
            sample['edges_v'] = sample['edges_v']/255.
            sample['edges_h'] = sample['edges_h'] / 255.
        if 'dist_h' in sample:
            sample['dist_v'] = sample['dist_v']/255.
            sample['dist_h'] = sample['dist_h'] / 255.


        sample['image_class'] = 1
        sample['label'] = sample['label']/self.args['labels_divide_by']
        sample['name'] = img_name
        sample['orig_size'] = self.orig_im_sizes[index]

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.img_paths[index]).convert('RGB')
        _target = Image.open(self.mask_paths[index])

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

    def transform_custom(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args['crop_size']),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __str__(self):
        return self.name+'(split=' + str(self.split) + ')'


