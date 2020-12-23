from utils.model_dataset_names import *
import torch
from metrics.evaluator import Evaluator

def get_standard_arguments(model_name, dataset_name, im_size):

    args = {}
    # args['data_dir'] = 'd:\\Olga'
    args['data_dir'] = 'c:\\users\\oveksler\\home'
    # args['data_dir'] = 'c:\\Home'

    args.update(get_dataset_args(args, dataset_name))   # gets argument related to datasets

    args['im_size'] = im_size
    args['base_size'] = args['im_size']
    args['crop_size'] = args['im_size']
    args['dataset_name'] = dataset_name

    args['use_fixed_features'] = True
    args['optimizer'] = 'Adam'

    args["momentum"] = 0.9
    args["weight_decay"] = 5e-4
    args["nesterov"] = False

    args['model_name'] = model_name
    args['main_metric'] = 'fscore'
    args['evaluator'] = Evaluator


    # scheduler related arguments
    args['scheduler'] = 'stepLR'
    args['scheduler_interval'] = 100
    args['scheduler_gamma'] = 0.1
    args['learning_rate'] = 0.001

    args['num_epoch_per_annealing_step'] = 1

    # various parameters for training with regularized loss, volume loss, etc.
    args['vol_min'] = 0.15
    args['vol_max'] = 0.85
    args['reg_weight'] = 100.
    args['vol_min_weight'] = 1.0
    args['vol_max_weight'] = 0.0
    args['vol_batch_weight'] = 1.0
    args['with_diag'] = False
    args['subtract_eps'] = 0.
    args['sigma'] = 0.15
    args['adaptive_sigma'] = False  #if true, compute sigma from each image by computing ave color variation
    args['neg_weight'] = 2.
    args['vol_min_smallest_weight'] = 1.
    args['middle_sq_loss_weight'] = 1.
    args['border_loss_weight'] = 1.

    # arguments for dense CRF
    args['sigma_rgb'] = 15.
    args['sigma_xy'] = 80.
    args['scale_factor'] = 1.

    args['verbose'] = True

    args['shuffle_train'] = True
    args['shuffle_test'] = False
    args['num_workers'] = 0
    args['train_batch_size'] = 16
    args['val_batch_size'] = 16

    args['valid_dataset'] = True
    args['save_best_loss_name'] = None
    args['visualize'] = True

    args['train_with_metric'] = True

    args['image_normalize_mean'] = (0.485, 0.456, 0.406)
    args['image_normalize_std'] = (0.229, 0.224, 0.225)
    args['negative_class'] = False

    args['split'] = 'train'

    args['save_stat'] = True

    args['final_activation'] = 'softmax'
    args['post_processing'] = None
    args['num_final_features'] = 64  #number of final features in Unet architectures

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # only one of the three below can be set to 'true'. None = False
    args['save_on_best_total_training_loss'] = True
    args['save_best_loss_name'] = None
    args['save_best_valid_metric'] = None

    args['sigma_xy'] = 50
    args['shifts'] = []

    args['radius'] = 5
    for x in range(args['radius']):
        for y in range(args['radius']):
            if x != 0 or y != 0:
                args['shifts'].append([x, y])
            if x  != 0 and y  != 0:
                args['shifts'].append([x, -y])


    args['read_edges'] = False
    args['read_dist'] = False
    args['read_colorquant'] = False

    args['colorquant_weight'] = 1000
    args['num_quantcolors'] = 17

    return args
