import torch
import torch.nn as nn


class GaussianLoss(nn.Module):
    __name__ = 'GaussianLoss'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.mean = param['mean']
        self.var = param['var']
        self.num_classes = param['num_classes']


    def forward(self, y_pr, y_gt, sample):

        if type(y_pr) is tuple:
            ft = y_pr[1]
            y_pr = y_pr[0]
        else:
            ft = y_pr


        comp = 3

        if comp == 1:
            n, c, h, w = y_pr.size()
            y_gt = y_gt.long()

            y_gt = y_gt.flatten()
            y_gt_mask = (y_gt >= 0) & (y_gt < c)

            y_gt = y_gt[y_gt_mask].long()

            II = torch.eye(c)
            one_hot = II[y_gt]

            one_hot = one_hot.to(self.device)

            y_pr = y_pr.transpose(1, 2)
            y_pr = y_pr.transpose(2, 3)
            y_pr = y_pr.reshape(n*w*h, c)
            y_pr = y_pr[y_gt_mask, :]

            n, c, h, w = ft.size()
            ft = ft.transpose(1, 2)
            ft = ft.transpose(2, 3)
            ft = ft.reshape(n*w*h, c)
            ft = ft[y_gt_mask, :]

            loss = y_pr * one_hot

            to_add = ft ** 2
            to_add = torch.sum(to_add)
            to_add = to_add / (2*self.var)

            loss = -torch.sum(loss)/y_pr.size()[0] + to_add

            return loss

        elif comp == 2:
            y_gt = y_gt.long()
            y_gt = y_gt.flatten()
            y_gt_mask = (y_gt >= 0) & (y_gt < self.num_classes)
            y_gt = y_gt[y_gt_mask].long()

            II = torch.eye(self.num_classes)
            one_hot = II[y_gt]

            one_hot = one_hot.to(self.device)

            n, c, h, w = ft.size()
            ft = ft.transpose(1, 2)
            ft = ft.transpose(2, 3).reshape(n * w * h, c)
            ft = ft[y_gt_mask, :]

            ft = ft.unsqueeze(dim=2)
            ft = ft.repeat(1, 1, self.num_classes)

            mean = torch.transpose(self.mean, 0, 1)
            mean = mean.unsqueeze(dim=0)

            loss = (ft - mean) ** 2

            var = torch.transpose(self.var, 0, 1)
            var = var.unsqueeze(dim=0)

            loss /= (2 * var)
            loss = torch.sum(loss, dim=1)

            loss = loss * one_hot

            loss = torch.sum(loss)
            return loss

        else:
            ft = torch.flatten(ft, start_dim=2)
            ft = ft.transpose(0, 1)
            ft = torch.flatten(ft, start_dim=1)
            ft = ft.transpose(0, 1)

            y_gt = torch.flatten(y_gt).long()

            loss = 0

            for cl in range(self.num_classes):
                current_class = (y_gt == cl)
                ft_cl = ft[current_class, :]
                if ft_cl.size()[0] == 0:
                    loss_cl = 0
                else:
                    mean_cl = self.mean[cl, :]
                    var_cl = self.var[cl, :]

                    loss_cl = (ft_cl - mean_cl) ** 2
                    loss_cl = loss_cl/(2 * var_cl)
                    loss_cl = torch.sum(loss_cl, dim=1)
                    loss_cl = torch.mean(loss_cl)
                loss = loss + loss_cl

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

