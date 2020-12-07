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



def test_model(model_name, dataset_name, im_size, model_load):

    args = get_standard_arguments(model_name, dataset_name, im_size)
    args['model_load'] = model_load
    # args['num_classes'] = 3
    # args['cats_dogs_separate'] = True

    _, args['loss_names'] = cross_entropy_loss(ignore_class=255)

    # _, args['loss_names'] = gaussian_loss(args)

    print("Testing input model")
    # args['split'] = 'train'
    test_logs = T.test(args, model_load)

if __name__ == "__main__":
    model_load = './run/experiments/single_class/OxfordPet_MobileNetV2_Ft_Linear_64__with_CE__V1.pt'
    test_model(model_name=model_names[0], dataset_name=singleclass_dataset_names[0], im_size=64, model_load=model_load)


