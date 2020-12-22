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
from utils.loss_sets import gaussian_loss, gaussian_loss_with_mixture, gauss_mixture_combined
from utils.get_dataset import *
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import matplotlib.pyplot as plt


def compute_means_and_var(args, model_load=None):
    evaluator = EvaluatorComputeMean

    args_local = copy.deepcopy(args)

    _, args_local['loss_names'] = gaussian_loss(args_local)

    args_local['split'] = 'train'

    #test_logs = test_on_correct_labels(args_local, model_load, evaluatorIn=evaluator)
    test_logs = T.test(args_local, model_load, EvaluatorIn=evaluator)

    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance']
    class_prob = test_logs['metrics']['class_prob']

    return mean, var, class_prob

def compute_pair(m1, m2, v1, v2):

    res = torch.abs(m1-m2)/(torch.sqrt(v1) + torch.sqrt(v2) + 0.000000001)
    return res

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

    #_, args['loss_names'] = gaussian_loss(args)
    #_, args['loss_names'] = gaussian_loss_with_mixture(args)
    _, args['loss_names'] = gauss_mixture_combined(args)

    for iter in range(num_iter):
        print('\nIteration: ', iter, ' mean of mean ', torch.mean(args['mean'], dim=1))
        print("Testing before changing mean")
        test_logs = T.test(args, model_load)

        args['old_mean'] = args['mean']

        if iter == 0:
            mean, var, class_prob = compute_means_and_var(args, model_load)

            args['var'] = var
            args['mean'] = mean
            args['class_prob'] = class_prob

        m1 = mean[0, :]
        m2 = mean[1, :]
        m3 = mean[2, :]

        v1 = var[0, :]
        v2 = var[1, :]
        v3 = var[2, :]

        # d12 = torch.sqrt(torch.sum((m1 - m2) ** 2))
        # d13 = torch.sqrt(torch.sum((m1 - m3) ** 2))
        # d23 = torch.sqrt(torch.sum((m3 - m2) ** 2))
        #

        d12 = compute_pair(m1, m2, v1, v2)
        d13 = compute_pair(m1, m3, v1, v3)
        d23 = compute_pair(m3, m2, v3, v2)

        sorted12, _ = torch.sort(d12)
        sorted13, _ = torch.sort(d13)
        sorted23, _ = torch.sort(d23)

        print(sorted12, 'median', sorted12[700])
        print(sorted13, 'median', sorted13[700])
        print(sorted23, 'median', sorted23[700])

        if iter > 0:
            pretrained_dict = torch.load(model_load)
            model = get_model(args)
            model_dict = model.state_dict()
            del pretrained_dict['mean']
            del pretrained_dict['sigma']
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            torch.save(model.state_dict(), model_load)

        print('\nupdated mean of mean ', torch.mean(mean, dim=1))
        print('\nmean of var', torch.mean(var, dim=1))

        print("Test after changing mean")
        test_logs = T.test(args, model_load)


        model_prev = get_model(args)

        if model_load is not None:
            model_prev.load_state_dict(torch.load(model_load))

        args['model_prev'] = model_prev.to(args['device'])
        model_prev.eval()
        #_, args['loss_names'] = gaussian_loss_with_mixture(args)
        _, args['loss_names'] = gauss_mixture_combined(args)

        valid_logs, train_logs, valid_metric = T.train_normal(args, num_epoch, model_save, model_load)
        model_load = model_save


        #print("\nALl the ds: d23, d12, d13 ", d23, d12, d13)


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
    args['shuffle_test'] = False

    one_stage_training_gauss(args, model_load)


def new_test():

    epsilon = torch.finfo(torch.float32).eps

    r = torch.rand([24])

    ft = torch.tensor(r).float()
    ft = ft.view(6, 4)
    num_classes = 3

    # mean = torch.rand([3,4])
    # var = torch.rand([3,4])
    # class_prob = torch.rand([3])
    # class_prob = class_prob/sum(class_prob)

    ft = torch.tensor([[0.0535, 0.7935, 0.2525, 0.3562],
            [0.7587, 0.0454, 0.1192, 0.0193],
            [0.5526, 0.1416, 0.7261, 0.2430],
            [0.1548, 0.0698, 0.1974, 0.7610],
            [0.8434, 0.5806, 0.4745, 0.6355],
            [0.3435, 0.8416, 0.3451, 0.8155]])
    mean = torch.tensor([[0.0213, 0.2804, 0.3500, 0.0705],
            [0.5077, 0.1460, 0.6972, 0.7937],
            [0.6053, 0.5292, 0.9267, 0.7868]])

    var = torch.tensor([[0.7306, 0.2218, 0.3051, 0.1479],
            [0.5749, 0.1398, 0.1326, 0.5127],
            [0.5174, 0.8227, 0.6534, 0.3736]])
    class_prob = torch.tensor([0.0836, 0.0315, 0.8849])

    n = 6

    print(ft)
    print(mean)
    print(var)
    print(class_prob)

    two_times_pi = 6.28318530718

    out = torch.zeros([n, num_classes])

    for cl in range(num_classes):
        mean_cl = mean[cl, :]
        var_cl = var[cl, :]

        next = (ft - mean_cl) ** 2
        next = next / (2 * var_cl)

        sigmas_cl = torch.sqrt(var_cl * two_times_pi)
        inside_exp = torch.log(sigmas_cl)

        next = next + inside_exp

        next = -next

        out[:, cl] = torch.sum(next, dim=1)

    out = out + torch.log(class_prob)
    max_val, _ = torch.max(out, dim=1, keepdim=True)

    out = out - max_val
    out = torch.exp(out)
    out = torch.sum(out, dim=1)
    out = torch.log(out + epsilon)
    max_val = torch.squeeze(max_val, dim=1)
    out = out + max_val
    out = torch.mean(out)
    print(out)


if __name__ == "__main__":

    #visualize_pca()

    train_gauss()

    #new_test()

    # model_load = './run/experiments/models/OxfordPet_MobileNetV2_Ft_Linear_128__with_CE__V2.pt'
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #
    # model_load = None
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #



