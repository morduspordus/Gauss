from __future__ import print_function, division
from .pascalSingleClass import PascalSingleClass

class PascalWebAugCombined(PascalSingleClass):
    """
        if  args.single_class = -1, extracts all classes
        if args.single_class > 0, extracts class given by args.single_class

    """
    __name__ = 'pascalWebAugCombined'
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param split: train
        """

        assert(args['negative_class'] == False) # cannot use this dataset with negative class

        PascalSingleClass.__init__(self, args, split)

        if split == 'train' or split == 'all':
            temp_img_paths = self.img_paths
            temp_mask_paths = self.mask_paths
            temp_img_names = self.img_names
            temp_true_class = self.true_class
            temp_orig_im_sizes = self.orig_im_sizes

            args['annotation_dir'] = args['annotation_dir_2']
            args['join_class_dir'] = args['join_class_dir_2']
            args['images_dir'] = args['images_dir_2']
            args['image_sets_dir'] =args['image_sets_dir_2']
            args['d_path'] = args['d_path_2']

            PascalSingleClass.__init__(self, args, split)
            self.img_paths.extend(temp_img_paths)
            self.mask_paths.extend(temp_mask_paths)
            self.img_names.extend(temp_img_names)
            self.true_class.extend(temp_true_class)
            self.orig_im_sizes.extend(temp_orig_im_sizes)

        args['dataset_name'] = 'pascalWebAugCombined'

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_paths)))

    def __str__(self):
        return 'pascalWebAugCombined(split=' + str(self.split) + ')'

    def name(self):
        return 'pascalWebAugCombined'

