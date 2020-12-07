from utils.model_dataset_names import model_classes


def get_model(args):

    if args['model_name'] in model_classes:
        model = model_classes[args['model_name']](args)
    else:
        print('Model {} not available.'.format(args['model_name']))
        raise NotImplementedError

    return model



