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
