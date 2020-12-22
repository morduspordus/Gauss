import losses.losses as lss

def cross_entropy_loss(ignore_class, weights=None):

    loss_names = {'cross_entropy': [lss.CrossEntropyLoss, {'ignore_class': ignore_class, 'weights': weights}]}

    return 'ce_loss', loss_names

#
# def gaussian_loss(args):
#
#     loss_names = {'gaussian_loss': [lss.GaussianLoss, {'num_classes': args['num_classes']}],
#                   'means_push': [lss.MeansPushLoss, {}]
#                   }
#
#     return 'gauss_loss', loss_names



def gaussian_loss(args):

    loss_names = {'gaussian_loss': [lss.GaussianLoss, {'num_classes': args['num_classes']}]
                  }

    return 'gauss_loss', loss_names


def gauss_mixture_combined(args):
    loss_names = {
                  'GaussMixtureCombined': [lss.GaussMixtureCombined, {'num_classes': args['num_classes']}]
                  }

    return 'combined', loss_names


def gaussian_loss_with_mixture(args):

    loss_names = {'gaussian_loss': [lss.GaussianLoss, {'num_classes': args['num_classes']}],
                  'mixture_loss':  [lss.MixtureLossFirstVersion, {'num_classes': args['num_classes']}]
                  }

    return 'gauss_loss', loss_names


# def gaussian_loss_with_mixture(args):
#
#     loss_names = {'gaussian_loss': [lss.GaussianLoss, {'num_classes': args['num_classes']}],
#                   'mixture_loss':  [lss.MixtureLossWithModelPrev, {'num_classes': args['num_classes'],
#                                     'model_prev': args['model_prev']}]
#
#                   }
#     return 'gauss_loss', loss_names