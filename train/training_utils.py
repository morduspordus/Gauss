import torch
from train.train import TrainEpoch, ValidEpoch
from utils.get_losses import get_losses
from utils.get_model import get_model
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler
from utils.file_utils import FileWrite
from utils.get_dataset import *
from utils.image_utils import visualize_results
from utils.other_utils import param_to_string, create_string_anneal
import numpy as np
from metrics.evaluator import Evaluator
import copy


def compute_pair(m1, m2, v1, v2):

    res = torch.abs(m1-m2)/(torch.sqrt(v1) + torch.sqrt(v2) + 0.000000001)
    return res


def analyze_mean_var(mean, var, sigma):

    neg_sigma = sigma < 0
    print("Negative sigma watch ", neg_sigma.any())

    m1 = mean[0, :]
    m2 = mean[1, :]
    m3 = mean[2, :]

    v1 = var[0, :]
    v2 = var[1, :]
    v3 = var[2, :]

    d12 = compute_pair(m1, m2, v1, v2)
    d13 = compute_pair(m1, m3, v1, v3)
    d23 = compute_pair(m3, m2, v3, v2)

    sorted12, _ = torch.sort(d12)
    sorted13, _ = torch.sort(d13)
    sorted23, _ = torch.sort(d23)

    print(sorted12.data, 'median', sorted12[700].data)
    print(sorted13.data, 'median', sorted13[700].data)
    print(sorted23.data, 'median', sorted23[700].data)

    print('\nmean of mean ', torch.mean(mean, dim=1).data)
    print('\nmean of var', torch.mean(var, dim=1).data, '\n')


def create_train_epoch_runner(model, evaluator, losses, optimizer, scheduler, args):

    train_epoch = TrainEpoch(
        model,
        evaluator=evaluator,
        losses=losses,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args
    )
    return train_epoch


def create_valid_epoch_runner(model, evaluator, losses, device, verbose, imageSaver):

    valid_epoch = ValidEpoch(
        model,
        evaluator=evaluator,
        losses=losses,
        device=device,
        verbose=verbose,
        imageSaver=imageSaver
    )

    return valid_epoch


def print_metrics(metrics):
    str_to_print = ''

    for m in metrics:
        if type(metrics[m]) is np.ndarray: # sometimes metrics are not single numbers, don't print
            temp = str(np.asscalar(metrics[m]))
            str_to_print = str_to_print + m + ': ' + temp + ', '

    print(str_to_print)


# if args.save_best_loss_name is not a None, saves model
# with the lowest validation loss whose name is given in args.save_best_loss_name
def train(model, optimizer, scheduler, losses, train_dataset, valid_dataset, num_epoch, args, evaluator, imgSaver=None):

    device = args['device']

    main_metric = evaluator.main_metric

    train_epoch = create_train_epoch_runner(model, evaluator, losses, optimizer, scheduler, args)
    train_loader = torch.utils.data.DataLoader(train_dataset, args['train_batch_size'], shuffle=args['shuffle_train'], num_workers=args['num_workers'])

    best_state_dict = None

    if args['valid_dataset']:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, args['val_batch_size'], shuffle=False, num_workers=args['num_workers'])
        valid_epoch = create_valid_epoch_runner(model, evaluator, losses, device, args['verbose'], imgSaver)

    max_score = 0.  #assumes for the main metric, higher is better
    best_loss = None
    best_loss_valid_metric = 0.
    valid_logs = None

    if args['visualize']:
        visualize_results(train_dataset, valid_dataset, model, device)

    for i in range(0, num_epoch):
        if args['verbose']:
            print('\nEpoch: {}/{}'.format(i+1, num_epoch))
            analyze_mean_var(model.mean, model.sigma ** 2, model.sigma)

        train_logs = train_epoch.run(train_loader)
        if args['verbose']:
            print_metrics(train_logs['metrics'])

        if args['valid_dataset']:
            valid_logs = valid_epoch.run(valid_loader)
            if args['verbose']:
                if args['verbose']:
                    print_metrics(valid_logs['metrics'])

            if max_score < valid_logs['metrics'][main_metric]:
                max_score = valid_logs['metrics'][main_metric]

                if args['save_best_valid_metric']:
                    best_state_dict = copy.deepcopy(model.state_dict())
                    if args['verbose']:
                        print("Saving model with better validation metric")

            if args['save_best_loss_name'] is not None:
                if best_loss is None:
                    best_loss = valid_logs[args['save_best_loss_name']]
                    best_loss_valid_metric = valid_logs['metrics'][main_metric]
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    if best_loss > valid_logs[args['save_best_loss_name']]:
                        best_loss = valid_logs[args['save_best_loss_name']]
                        best_state_dict = copy.deepcopy(model.state_dict())
                        best_loss_valid_metric = valid_logs['metrics'][main_metric]
                        print("Found better")

        if args['save_on_best_total_training_loss']:
            assert(args['save_best_loss_name'] is None) # can't save based on both criterions
            if best_loss is None:
                best_loss = train_logs['total_loss']
                best_state_dict = copy.deepcopy(model.state_dict())
                if args['valid_dataset']:
                    best_loss_valid_metric = valid_logs['metrics'][main_metric]

            else:
                if best_loss > train_logs['total_loss']:
                    best_loss = train_logs['total_loss']
                    best_state_dict = copy.deepcopy(model.state_dict())
                    print("Found better loss")
                    if args['valid_dataset']:
                        best_loss_valid_metric = valid_logs['metrics'][main_metric]

        if args['visualize']:
            visualize_results(train_dataset, valid_dataset, model, device)

    if args['verbose']:
        print("\nMax metric score: {}".format(max_score))

    if args['save_best_valid_metric']:

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        if args['verbose']:
            print("\nSaving Model with best valid metric, it has val metric {}".format(max_score))

    if args['save_best_loss_name'] is not None:
        model.load_state_dict(best_state_dict)
        if args['verbose']:
          print("\nSaving Model with lowest Loss, it has val metric {}".format(best_loss_valid_metric))

    if args['save_on_best_total_training_loss']:
        model.load_state_dict(best_state_dict)
        if args['verbose']:
          print("\nSaving Model with lowest Loss. ")
          if args['valid_dataset']:
              print("It has metric ",  best_loss_valid_metric)

    return valid_logs, train_logs, best_loss_valid_metric


def train_normal(args, num_epoch, model_save=None, model_load=None, imgSaver=None, EvaluatorIn=None):
    # model_save is the name of the file name to save model to
    # model_load is the model to initialize training with

    if EvaluatorIn is None:
        evaluator = Evaluator(args)
    else:
        evaluator = EvaluatorIn(args)

    model = get_model(args)
    losses = get_losses(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    valid_metric = None
    main_metric = evaluator.main_metric

    if model_load is not None:
        model.load_state_dict(torch.load(model_load), strict=False)

    train_dataset, valid_dataset = get_train_val_datasets(args)

    valid_logs, train_logs, best_loss_valid_metric = train(model, optimizer, scheduler, losses, train_dataset,
                                                           valid_dataset, num_epoch, args, evaluator, imgSaver)

    if args['valid_dataset'] and args['train_with_metric']:
        valid_metric = valid_logs['metrics'][main_metric]

    if args['save_best_loss_name'] is not None or args['save_on_best_total_training_loss']:
        valid_metric = best_loss_valid_metric

    if not args['valid_dataset']:
        valid_metric = train_logs['metrics'][main_metric]
        valid_logs = train_logs

    if model_save is not None:
        torch.save(model.state_dict(), model_save)

    return valid_logs, train_logs, valid_metric


def train_anneal(args, model_save, model_load, EvaluatorIn=None, imgSaver=None):
    # model_save is the name of the file name to save model to
    # model_load is the model to initialize training with

    fw = FileWrite(args['file_stat_name'], args['save_stat'])

    if EvaluatorIn is None:
        evaluator = Evaluator(args)
    else:
        evaluator = EvaluatorIn(args)

    main_metric = evaluator.main_metric

    num_epochs_per_step = [args['num_epoch_per_annealing_step']] * len(args['reg_loss_params'])

    for i, param in zip(range(len(num_epochs_per_step)), args['reg_loss_params']):
        old_params = args['loss_names'][args['reg_loss_name']][1]

        for k, v in param.items():
            if k in old_params: old_params[k] = param[k]


        args['loss_names'][args['reg_loss_name']][1] = old_params

        if args['verbose']:
            print("\nOuter epoch {}/{}: params: {}".format(i+1, len(num_epochs_per_step), param_to_string(param)))

        valid_logs, train_logs, valid_metric = train_normal(args, num_epochs_per_step[i], model_save, model_load,
                                                            imgSaver, EvaluatorIn)
        model_load = model_save

        train_metric = train_logs['metrics'][main_metric]

        str_to_write = create_string_anneal(param, train_metric, valid_metric)

        fw.write_to_file(str_to_write)

    return train_logs


def test(args, model_load, imageSaver=None, EvaluatorIn=None):

    model = get_model(args)

    if model_load is not None:
        model.load_state_dict(torch.load(model_load), strict=False)

    logs = test_with_loaded_model(args, model, imageSaver, EvaluatorIn)

    return logs

def test_with_loaded_model(args, model, imageSaver=None, EvaluatorIn=None):
    device = args['device']

    losses = get_losses(args)

    if EvaluatorIn is None:
        evaluator = Evaluator(args)
    else:
        evaluator = EvaluatorIn(args)

    valid_dataset = get_val_dataset(args, args['split'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, args['val_batch_size'], shuffle=args['shuffle_test'], num_workers=args['num_workers'])

    valid_epoch = create_valid_epoch_runner(model,
                                            evaluator,
                                            losses,
                                            device,
                                            args['verbose'],
                                            imageSaver=imageSaver)

    logs = valid_epoch.run(valid_loader)
    print_metrics(logs['metrics'])

    return logs


