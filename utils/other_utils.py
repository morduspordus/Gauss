import sys
from utils.file_utils import FileWrite
import os
import random
import numpy as np
import torch



class Logger(object):
    def __init__(self,file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w",1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def create_string(valid_logs, dataset_name, model_name, kernel, image_height):
    str_to_return = "\ndataset: {} model: {} kernel: {} img_h: {}".format(dataset_name, model_name, kernel, image_height)

    str_logs = ['{} - {:.4}'.format(k, v) for k, v in valid_logs.items()]
    str_logs = ', '.join(str_logs)

    return str_to_return + ' ::: ' + str_logs



def create_string_pascal(valid_logs, cl, len_train=None, len_val=None):
    str_to_return = "\nclass: {} class_name: {} ".format(cl,pascal_class_names[cl])

    if len_train is not None:
        str_to_return = str_to_return+ "len_train: {}".format(len_train)
    if len_train is not None:
        str_to_return = str_to_return+ "len_val: {}".format(len_val)

    str_logs = ['{} - {:.4}'.format(k, v) for k, v in valid_logs.items()]
    str_logs = ', '.join(str_logs)

    return str_to_return + ' ::: ' + str_logs



def param_to_string(param):
    str_params = ['{} - {:.4}'.format(k, v) for k, v in param.items()]
    s = ', '.join(str_params)
    return s



def create_string_anneal(param, train_metric, val_metric):
    str_to_return = "train_metric: {}  val_metric: {}".format(train_metric, val_metric)

    str_params = param_to_string(param)

    return str_params + ':::' + str_to_return + '\n'




def model_file_name(dataset_name, model_name, valid_metric, model_load, model_params, dataset_params):

    smooth_kernel = model_params["smooth_kernel"]
    image_height = dataset_params["image_height"]
    valid_metric = "{0:.2f}".format(valid_metric * 100)

    if model_load == None:
        from_model = "None"
    elif model_load == "Anneal":
        from_model = "Anneal"
    else:
        x = model_load.split("\\")
        x = x[2]
        y = x.split('.pt')
        from_model = y[0]

    if len(from_model) > 40:
        from_model = from_model[:40]

    if smooth_kernel == None:
        kernel_str = "_nosmooth_"
    else:
        kernel_str ="_smooth" + str(smooth_kernel)+"_"

    model_weights_name = os.path.join(WEIGHTS_DIR,  dataset_name  +str(image_height)+"_" + model_name + kernel_str + '_M_' + valid_metric +  '_FFOM_' + from_model + '.pt')

    return model_weights_name



def create_string_pascal(valid_logs, cl, len_train=None, len_val=None):
    str_to_return = "\nclass: {} class_name: {} ".format(cl,pascal_class_names[cl])

    if len_train is not None:
        str_to_return = str_to_return+ "len_train: {}".format(len_train)
    if len_train is not None:
        str_to_return = str_to_return+ "len_val: {}".format(len_val)

    str_logs = ['{} - {:.4}'.format(k, v) for k, v in valid_logs.items()]
    str_logs = ', '.join(str_logs)

    return str_to_return + ' ::: ' + str_logs


def create_string_from_logs(valid_logs, valid_metric):

    str_to_return = "\nvalid metric: {}".format(valid_metric)

    str_logs = ['{} - {}'.format(k, v) for k, v in valid_logs.items()]
    str_logs = ', '.join(str_logs)

    return str_to_return + ' ::: ' + str_logs


def create_string_from_args(args):

    str_args = ['{} - {}'.format(k, v) for k, v in args.items()]
    str_args = '\n'.join(str_args)

    return str_args

def create_file_name(dataset_name, model_name, im_size, training_type, output_dir):

    version = 1
    add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
    model_save = os.path.join(output_dir, add_to_file_path + '.pt')

    while os.path.exists(model_save):
        version += 1
        add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
        model_save = os.path.join(output_dir, add_to_file_path + '.pt')

    return add_to_file_path, model_save

def create_dir_name(dataset_name, model_name, im_size, training_type, output_dir):

    version = 1
    add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
    model_save = os.path.join(output_dir, add_to_file_path)

    while os.path.exists(model_save):
        version += 1
        add_to_file_path = dataset_name + '_' + model_name + '_' + str(im_size) + '_' + training_type + '_V' + str(version)
        model_save = os.path.join(output_dir, add_to_file_path)

    return model_save


def save_args_outcome(add_to_file_path, args, valid_logs, valid_metric, output_dir):

    file_args_name = os.path.join(output_dir, add_to_file_path + '_args.txt')
    fw = FileWrite(file_args_name, True)
    str_to_write = create_string_from_args(args)
    fw.write_to_file(str_to_write)

    file_outcome_name = os.path.join(output_dir, add_to_file_path + '_outcome.txt')
    fw = FileWrite(file_outcome_name, True)
    str_to_write = create_string_from_logs(valid_logs, valid_metric)
    fw.write_to_file(str_to_write)


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, use_cudnn_benchmark):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    if use_cudnn_benchmark:
        print("We will use `torch.backends.cudnn.benchmark`")
    else:
        print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


def reg_weight_list(start_reg_weight, num_iter, reg_weight_increment):

    return list(range(start_reg_weight, start_reg_weight + num_iter * reg_weight_increment,
                      reg_weight_increment))
