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
from utils.loss_sets import gaussian_loss, gaussian_loss_with_mixture, gauss_mixture_combined



def compute_means_and_var(args, model_load=None):
    evaluator = EvaluatorComputeMean

    args_local = copy.deepcopy(args)

    _, args_local['loss_names'] = gaussian_loss(args_local)

    args_local['split'] = 'train'

    test_logs = T.test(args_local, model_load, EvaluatorIn=evaluator)

    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance']
    class_prob = test_logs['metrics']['class_prob']

    return mean, var, class_prob

def one_stage_training_gauss(args, model_load):

    training_type = '_gauss_'
    output_dir = './run/experiments/models'
    add_to_file_path, model_save = create_file_name(args['dataset_name'], args['model_name'], args['im_size'], training_type, output_dir)

    num_iter = 100
    num_epoch = 1

    args['num_epoch'] = num_epoch

    args['use_fixed_features'] = False
    args['mean_requires_grad'] = True

    args['learning_rate'] = 0.00001

    args['mean'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    args['var'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    args['class_prob'] = torch.zeros(args['num_classes']) + 1/args['num_classes']

    _, args['loss_names'] = gauss_mixture_combined(args)

    change_mean_from_gauss_first_iter = True
    change_mean_from_gauss_other_iter = True

    for iter in range(num_iter):
        print('\nIteration: ', iter, '****************************************')
        print("Testing current model")
        test_logs = T.test(args, model_load)

        if (change_mean_from_gauss_first_iter and iter == 0):
            mean, var, class_prob = compute_means_and_var(args, model_load)

            args['var'] = var
            args['mean'] = mean
            args['class_prob'] = class_prob

            print("Test after changing mean before any iterations")
            test_logs = T.test(args, model_load)

        if iter > 0 and change_mean_from_gauss_other_iter:
            mean, var, class_prob = compute_means_and_var(args, model_load)

            args['var'] = var
            args['mean'] = mean
            args['class_prob'] = class_prob

            pretrained_dict = torch.load(model_load)
            model = get_model(args)
            model_dict = model.state_dict()

            # this ensures means and variances stay as specified in args[mean] and args[var]
            del pretrained_dict['mean']
            del pretrained_dict['sigma']

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            torch.save(model.state_dict(), model_load)

            print("Test after changing mean")
            test_logs = T.test(args, model_load)


        _, args['loss_names'] = gauss_mixture_combined(args)

        T.train_normal(args, num_epoch, model_save, model_load)
        model_load = model_save



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

    args['train_batch_size'] = 2
    args['val_batch_size'] = 8
    args['main_metric'] = 'miou'

    one_stage_training_gauss(args, model_load)


if __name__ == "__main__":

    train_gauss()