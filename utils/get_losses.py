def get_losses(args):

    losses = []
    names_dict = args['loss_names']

    for name, other in names_dict.items():

        loss_class = other[0]
        param = other[1]

        param['device'] = args['device']

        if 'negative_class_loss' in names_dict:
            param['negative_class'] = True
        else:
            param['negative_class'] = False

        losses.append(loss_class(param))

    return losses
