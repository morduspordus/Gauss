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


def one_stage_training(model_name, dataset_name, im_size, model_load):

    args = get_standard_arguments(model_name, dataset_name, im_size)

    one_stage_training_with_args(model_name, dataset_name, im_size, model_load, args)

def one_stage_training_with_args(model_name, dataset_name, im_size, model_load, args):

    training_type = '_with_CE_change_feat_'

    output_dir = './run/experiments/models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    add_to_file_path, model_save = create_file_name(dataset_name, model_name, im_size, training_type, output_dir)

    num_epoch = 100
    args['use_fixed_features'] = False
    args['num_epoch'] = num_epoch
    args['model_load'] = model_load
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True
    args['num_features'] = 1536
    args['learning_rate'] = 0.0001
    # args['mean_requires_grad'] = True
    # args['mean'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    # args['var'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    # args['train_batch_size'] = 4
    # args['val_batch_size'] = 8

    #args['use_fixed_features'] = False
    #args['train_batch_size'] = 8

    _, args['loss_names'] = cross_entropy_loss(ignore_class=255)

    if model_load is not None:
        print("Testing input model")
        test_logs = T.test(args, model_load)

    valid_logs, train_logs, valid_metric = T.train_normal(args, num_epoch, model_save, model_load)

    save_args_outcome(add_to_file_path, args, valid_logs, valid_metric, output_dir)

if __name__ == "__main__":
    
    model_load = None

    one_stage_training(model_name=model_names[1],
                       dataset_name=singleclass_dataset_names[0],
                       im_size=128,
                       model_load=model_load)

