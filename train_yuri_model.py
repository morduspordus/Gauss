"""
    Trains on specified dataset, model, with annealing
    Possible datasets: 'pascal', 'oxfordpet', 'msra_b'
"""
import os
import train.training_utils as T
from utils.model_dataset_names import model_names, singleclass_dataset_names
from utils.get_standard_arguments import get_standard_arguments
from utils.file_utils import FileWrite
from utils.other_utils import create_string_from_logs, create_string_from_args, create_file_name, save_args_outcome
from utils.get_losses import *
import copy
from utils.loss_sets import cross_entropy_loss
from metrics.evaluator_yuri import EvaluatorComputeMean
from metrics.evaluator import Evaluator
from utils.imageSaver import ImageSaver
from dataloaders.utils import decode_segmap_pascal
import torch
from train.train import TrainEpoch, ValidEpoch
from utils.get_losses import get_losses
from utils.get_model import get_model
from utils.get_dataset import get_val_dataset
from train.training_utils import test_with_loaded_model
import sys
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils.loss_sets import gaussian_loss
from utils.get_dataset import *
from segmentation_models_pytorch.utils.meter import AverageValueMeter


def compute_means_and_var(args, model_load=None):
    evaluator = EvaluatorComputeMean

    args_local = copy.deepcopy(args)

    _, args_local['loss_names'] = gaussian_loss(args_local)

    args_local['split'] = 'train'

    #test_logs = test_on_correct_labels(args_local, model_load, evaluatorIn=evaluator)
    test_logs = T.test(args_local, model_load, EvaluatorIn=evaluator)

    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance']

    return mean, var


def one_stage_training_gauss(args, model_load):

    training_type = '_gauss_'

    output_dir = './run/experiments/models'

    add_to_file_path, model_save = create_file_name(args['dataset_name'], args['model_name'], args['im_size'], training_type, output_dir)

    num_iter = 100
    num_epoch = 1

    args['num_epoch'] = num_epoch

    args['use_fixed_features'] = False
    args['mean_requires_grad'] = False

    args['learning_rate'] = 0.000001

    args['mean'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    args['var'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization

    _, args['loss_names'] = gaussian_loss(args)

    for iter in range(num_iter):
        print('\nIteration: ', iter, ' mean of mean ', torch.mean(args['mean'], dim=1))
        print("Testing before changing mean")
        test_logs = T.test(args, model_load)

        mean, var = compute_means_and_var(args, model_load)
        args['mean'] = mean
        args['var'] = var

        if iter > 0:
            pretrained_dict = torch.load(model_load)
            model = get_model(args)
            model_dict = model.state_dict()
            del pretrained_dict['mean']
            del pretrained_dict['var']
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            torch.save(model.state_dict(), model_load)

        print('\nupdated mean of mean ', torch.mean(mean, dim=1))
        print('\nmean of var', torch.mean(var, dim=1))

        print("Test after changing mean")
        test_logs = T.test(args, model_load)

        valid_logs, train_logs, valid_metric = T.train_normal(args, num_epoch, model_save, model_load)
        model_load = model_save

        m1 = mean[0, :]
        m2 = mean[1, :]
        m3 = mean[2, :]

        d12 = torch.sqrt(torch.sum((m1 - m2) ** 2))
        d13 = torch.sqrt(torch.sum((m1 - m3) ** 2))
        d23 = torch.sqrt(torch.sum((m3 - m2) ** 2))

        print("\nALl the ds: d23, d12, d13 ", d23, d12, d13)


def train_gauss():
    #model_load = './run/experiments/models/OxfordPet_MobileNetV2_Ft_Linear_128__with_CE_change_feat__V3.pt'
    model_load = None
    model_name = model_names[2]
    dataset_name = singleclass_dataset_names[0]
    im_size = 128

    args = get_standard_arguments(model_name, dataset_name, im_size)

    args['num_features'] = 1536
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True

    args['train_batch_size'] = 4
    args['val_batch_size'] = 8
    args['main_metric'] = 'miou'

    one_stage_training_gauss(args, model_load)


if __name__ == "__main__":

    train_gauss()
    #temp_test()

    # model_load = './run/experiments/models/OxfordPet_MobileNetV2_Ft_Linear_128__with_CE__V2.pt'
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #
    # model_load = None
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #



