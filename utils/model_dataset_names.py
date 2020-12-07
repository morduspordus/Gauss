import os
from dataloaders.datasets.oxford_iii_pet import *
from dataloaders.datasets.SaliencyCommon import *
from dataloaders.datasets.pascalSingleClass import *
from dataloaders.datasets.pascalWebAugCombined import *
from dataloaders.datasets.pascal import *
from models.Unet.unet_fixed_features import *

model_names = ["MobileNetV2_Ft", "MobileNetV2_Ft_Linear"]


model_classes = {"MobileNetV2_Ft": MobileNetV2_Ft, "MobileNetV2_Ft_Linear": MobileNetV2_Ft_Linear}

singleclass_dataset_names = ['OxfordPet', 'MSRA',  'ECSSD', 'DUT', 'Diningtable', 'DUTS', 'SED2', 'HKU_IS', 'Thur', 'PascalS', 'SOD']
pascal_multi_dataset_names = ['pascal']


def give_data_dir():
    # root directory where data is located
    return 'd:\\Olga'


def get_dataset_args(args, dataset):
    data_dir = args['data_dir']

    to_return = {}
    to_return['labels_divide_by'] = 1.  # divide labels by this

    if dataset == singleclass_dataset_names[0]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Oxford_iit_pet')  # dir where dataset is located
        to_return['d_class'] = OxfordPet
        to_return['ignore_class'] = 255
        to_return['cats_only'] = False
        to_return['dogs_only'] = False
        to_return['cats_dogs_separate'] = False
        to_return['dogs_negative'] = False
        to_return['cats_negative'] = False
        to_return['num_classes'] = 2

    elif dataset == singleclass_dataset_names[1]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/MSRA_B')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[2]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/ECSSD')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[3]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/DUT-OMRON')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[4]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/11diningtable')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['add_file_extension'] = True
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[5]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/DUTS')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[6]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/SED2')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 1

    elif dataset == singleclass_dataset_names[7]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/HKU_IS')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[8]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/Thur')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[9]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/PascalS')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 255

    elif dataset == singleclass_dataset_names[10]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/Saliency/SOD')
        to_return['d_class'] = SaliencyCommon
        to_return['ignore_class'] = -1
        to_return['num_classes'] = 2
        to_return['labels_divide_by'] = 1


    elif dataset == pascal_multi_dataset_names[0]:
        to_return['d_path'] = os.path.join(data_dir, 'Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/')
        to_return['d_class'] = VOCSegmentation
        to_return['num_classes'] = 21
        to_return['ignore_class'] = 255
        to_return['use_augmentation'] = True


    else:
        print('Dataset {} not available.'.format(dataset))
        raise NotImplementedError

    to_return['data_dir'] = data_dir

    return to_return

