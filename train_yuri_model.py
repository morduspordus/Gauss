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
from utils.loss_sets import gaussian_loss


def test_with_matrix(args, model_load, imageSaver=None, EvaluatorIn=None, ft_matrix=None):

    training_type = 'with_blah'
    model = get_model(args)

    if model_load is not None:
        model.load_state_dict(torch.load(model_load))

    num_classes, num_features = ft_matrix.size()
    num_features = num_features - 1

    to_replace = ft_matrix[:, num_features:].flatten()
    model.conv_layer.bias[:] = to_replace

    to_replace = ft_matrix[:, 0:num_features]
    to_replace = to_replace.unsqueeze(dim=2)
    to_replace = to_replace.unsqueeze(dim=2)

    model.conv_layer.weight.data = to_replace

    output_dir = './run/experiments/models'

    dataset_name = "OxfordPet"
    model_name = "yuri"
    im_size = 128
    add_to_file_path, model_save = create_file_name(dataset_name, model_name, im_size, training_type, output_dir)

    logs = test_with_loaded_model(args, model, imageSaver, EvaluatorIn)
    torch.save(model.state_dict(), model_save)

    return logs


def my_own_test(ft_matrix, args, evaluatorIn, model_load):
    valid_dataset = get_val_dataset(args, args['split'])
    dataloader = torch.utils.data.DataLoader(valid_dataset, args['val_batch_size'], shuffle=False,
                                               num_workers=args['num_workers'])

    verbose = args['verbose']
    device = args['device']

    evaluator = evaluatorIn(args)
    evaluator.reset()
    ft_matrix = ft_matrix.to(device)
    ft_matrix = torch.transpose(ft_matrix, 0, 1)
    model = get_model(args)

    if model_load is not None:
        model.load_state_dict(torch.load(model_load))

    model.to(device)
    model.eval()


    with tqdm(dataloader, desc='test', file=sys.stdout, disable=not (verbose)) as iterator:
        for sample in iterator:
            x = sample['image']
            y = sample['label']

            x, y = x.to(device), y.to(device)
            out = model.forward(x)

            if type(out) == tuple:
                ft = out[1]
            else:
                ft = out

            [n, d, h, w] = list(ft.size())
            ft = torch.cat([ft, torch.ones(n, 1, h, w).to(device)], 1)
            #ft = torch.nn.functional.normalize(ft, dim=1, p=2)

            ft = torch.transpose(ft, 0, 1)
            ft = torch.flatten(ft, start_dim=1)
            ft = torch.transpose(ft, 0, 1)

            res = torch.matmul(ft, ft_matrix)
            #res = torch.matmul(ft, torch.transpose(ft_matrix,0,1))
            _, pred = torch.max(res, dim=1)

            pred = pred.view(n, h, w)

            evaluator.add_batch(y, pred)

    results = evaluator.compute_all_metrics()
    print(results)


def compute_means_and_test(model_name, dataset_name, im_size, model_load):

    args = get_standard_arguments(model_name, dataset_name, im_size)

    args['num_features'] = 1539
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True

    args['train_batch_size'] = 8
    args['val_batch_size'] = 8

    evaluator = EvaluatorComputeMean

    _, args['loss_names'] = cross_entropy_loss(ignore_class=255)
    args['split'] = 'train'
    test_logs = T.test(args, model_load, EvaluatorIn=evaluator)

    ft_matrix = test_logs['metrics']['feature_sum']
    print(ft_matrix)
    print(ft_matrix[:, 1539])

    images_dir = './run/Temp/images'
    #imgSaver = ImageSaver(args, images_dir, out_multiplier=1, decode_segmap=decode_segmap_pascal,
    #                      with_gt=True, with_orig_im=True, with_orig_size=True)

    args['split'] = 'val'
    imgSaver = None
    # test_logs = test_with_matrix(args, model_load, imageSaver=imgSaver, ft_matrix=ft_matrix)
    # my_own_test(ft_matrix, args, evaluatorIn=Evaluator, model_load=model_load)
    # #
    # # Do gauss test
    epsilon = 0.000001
    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance'] + epsilon
    class_sizes = ft_matrix[:, args['num_features']]
    sum = torch.sum(class_sizes)
    class_pr = class_sizes/sum

    ft_matrix = mean/var
    mean_by_var = -mean/(2*var)
    mean_t = torch.transpose(mean, 0, 1)
    bias = torch.matmul(mean_by_var, mean_t)
    diag = torch.diagonal(bias)

    # nll = -torch.log(class_pr)
    # diag = diag + nll

    diag = torch.unsqueeze(diag, dim=1)
    ft_matrix = torch.cat((ft_matrix, diag), dim=1)
    test_logs = test_with_matrix(args, model_load, imageSaver=imgSaver, ft_matrix=ft_matrix)
    #my_own_test(ft_matrix, args, evaluatorIn=Evaluator, model_load=model_load)


def compute_means_and_var(args, model_load=None):

    evaluator = EvaluatorComputeMean

    #_, args['loss_names'] = cross_entropy_loss(ignore_class=255)
    _, args['loss_names'] = gaussian_loss(args)

    args['split'] = 'train'

    test_logs = T.test(args, model_load, EvaluatorIn=evaluator)

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

    args['print_mean'] = False

    args['use_fixed_features'] = False
    args['mean_requires_grad'] = True

    args['learning_rate'] = 0.0001

    args['mean'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    args['var'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    _, args['loss_names'] = gaussian_loss(args)

    for iter in range(num_iter):
        print("Testing before changing mean")
        test_logs = T.test(args, model_load)

        mean, var = compute_means_and_var(args, model_load)
        args['mean'] = mean
        args['var'] = var
        print('mean', mean)
        print('var', var)

        _, args['loss_names'] = gaussian_loss(args)

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
    model_load = None
    model_name = model_names[2]
    dataset_name = singleclass_dataset_names[0]
    im_size = 64

    args = get_standard_arguments(model_name, dataset_name, im_size)

    args['num_features'] = 1539
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True


    args['train_batch_size'] = 8
    args['val_batch_size'] = 8

    one_stage_training_gauss(args, model_load)


if __name__ == "__main__":

    train_gauss()

    # model_load = './run/experiments/models/OxfordPet_MobileNetV2_Ft_Linear_128__with_CE__V2.pt'
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #
    # model_load = None
    # compute_means_and_test(model_name=model_names[1], dataset_name=singleclass_dataset_names[0], im_size=128, model_load=model_load)
    #



