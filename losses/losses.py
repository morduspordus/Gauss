import torch
import torch.nn as nn


class GaussianLoss(nn.Module):
    __name__ = 'GaussianLoss'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.num_classes = param['num_classes']


    def forward(self, y_pr, y_gt, sample):

        ft = y_pr[1]
        mean = y_pr[2]
        var = y_pr[3]
        y_pr = y_pr[0]

        # ft = torch.flatten(ft, start_dim=2)
        # ft = ft.transpose(0, 1)
        # ft = torch.flatten(ft, start_dim=1)
        # ft = ft.transpose(0, 1)

        y_gt = torch.flatten(y_gt).long()

        loss = 0

        for cl in range(self.num_classes):
            current_class = (y_gt == cl)
            ft_cl = ft[current_class, :]
            if ft_cl.size()[0] == 0:
                loss_cl = 0
            else:
                mean_cl = mean[cl, :]
                var_cl = var[cl, :]

                loss_cl = (ft_cl - mean_cl) ** 2
                loss_cl = loss_cl/(2 * var_cl)
                loss_cl = torch.sum(loss_cl, dim=1)
                loss_cl = torch.mean(loss_cl)
            loss = loss + loss_cl

        loss = loss - (1/3) * (torch.mean( (mean[0,:] - mean[1,:]) **2)  + \
                      torch.mean( (mean[0,:] - mean[2,:]) **2) + \
                      torch.mean((mean[2, :] - mean[1, :]) ** 2))

        return loss


class CrossEntropyLoss(nn.Module):
    __name__ = 'CrossEntropyLoss'

    def __init__(self, param):
        super().__init__()
        self.ignore_class = param['ignore_class']
        self.weights = param['weights']

        if self.weights is None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_class)
        else:
            weights = torch.Tensor(self.weights)
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_class, weight=weights)

    def forward(self, y_pr, y_gt, sample):
        y_gt = y_gt.long()

        if type(y_pr) is tuple:
            y_pr = y_pr[0]

        return self.ce_loss(y_pr, y_gt)

