import torch
import torch.nn as nn
from models.Unet.unet_fixed_features import compute_neg_log_lk


class MixtureLossWithModelPrev(nn.Module):
    __name__ = 'MixtureLossWithModelPrev'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.num_classes = param['num_classes']
        self.epsilon = torch.finfo(torch.float32).eps
        self.two_times_pi = 6.28318530718
        self.model_prev = param['model_prev']

    def forward(self, y_pr, y_gt, sample):

        ft = y_pr[1]
        mean = y_pr[2]
        var = y_pr[3]
        class_prob = y_pr[4]
        x = sample['image']

        prediction = self.model_prev.forward(x.to(self.device))
        ft_old = prediction[1]

        [n, num_features] = ft.size()
        var = var + self.epsilon

        out = torch.zeros([n, self.num_classes]).to(self.device)

        for cl in range(self.num_classes):
            mean_cl = mean[cl, :]
            var_cl = var[cl, :]

            loss_cl = -compute_neg_log_lk(ft_old, mean_cl, var_cl)
            out[:, cl] = loss_cl

        soft_membership = torch.zeros([n, self.num_classes]).to(self.device)
        for cl in range(self.num_classes):
            bottom = 1.
            for cl_other in range(self.num_classes):
                if cl_other != cl:
                    bottom = bottom + (class_prob[cl_other]/class_prob[cl]) * torch.exp(out[:, cl_other] - out[:, cl])
            soft_membership[:, cl] = 1./bottom

        for cl in range(self.num_classes):
            mean_cl = mean[cl, :]
            var_cl = var[cl, :]

            loss_cl = -compute_neg_log_lk(ft, mean_cl, var_cl)
            loss_cl = loss_cl + torch.log(class_prob[cl])
            out[:, cl] = loss_cl

        out = out * soft_membership

        return torch.mean(out)


class MixtureLossFirstVersion(nn.Module):
    # first enropy verion I tried
    __name__ = 'MixtureLossFirstV'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.num_classes = param['num_classes']
        self.epsilon = torch.finfo(torch.float32).eps
        self.two_times_pi = 6.28318530718

    def forward(self, y_pr, y_gt, sample):

        ft = y_pr[1]
        mean = y_pr[2]
        var = y_pr[3]
        class_prob = y_pr[4]

        [n, num_features] = ft.size()
        var = var + self.epsilon

        out = torch.zeros([n, self.num_classes]).to(self.device)

        for cl in range(self.num_classes):
            mean_cl = mean[cl, :]
            var_cl = var[cl, :]

            next = (ft - mean_cl) ** 2
            next = next / (2 * var_cl)

            sigmas_cl = torch.sqrt(var_cl * self.two_times_pi)
            inside_exp = torch.log(sigmas_cl)

            next = next + inside_exp

            next = -next

            out[:, cl] = torch.sum(next, dim=1)

        out = out + torch.log(class_prob.to(self.device))
        max_val, _ = torch.max(out, dim=1, keepdim=True)

        out = out - max_val
        out = torch.exp(out)
        out = torch.sum(out, dim=1)
        out = torch.log(out + self.epsilon)
        max_val = torch.squeeze(max_val, dim=1)
        out = out + max_val
        out = torch.mean(out)

        return out

class GaussianLoss(nn.Module):
    __name__ = 'GaussianLoss'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.num_classes = param['num_classes']
        self.epsilon = torch.finfo(torch.float32).eps
        self.sqrt_2_times_pi = 2.50662827463

    def forward(self, y_pr, y_gt, sample):

        ft = y_pr[1]
        mean = y_pr[2]
        var = y_pr[3]
        class_prob = y_pr[4]

        y_gt = torch.flatten(y_gt).long()

        loss = 0

        [n, c] = ft.size()

        for cl in range(self.num_classes):
            current_class = (y_gt == cl)
            ft_cl = ft[current_class, :]
            if ft_cl.size()[0] == 0:
                loss_cl = 0.
            else:
                mean_cl = mean[cl, :]
                var_cl = var[cl, :]

                loss_cl = torch.sum(compute_neg_log_lk(ft_cl, mean_cl, var_cl) - torch.log(class_prob[cl]))
            loss = loss + loss_cl

        loss = loss/n

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



class MeansPushLoss(nn.Module):
    __name__ = 'MeansPush'

    def __init__(self, param):
        super().__init__()

    def forward(self, y_pr, y_gt, sample):

        mean = y_pr[2]

        loss = -1000 * (torch.mean( (mean[0,:] - mean[1,:]) **2)  + \
                      torch.mean((mean[0,:] - mean[2,:]) **2) + \
                      torch.mean((mean[2, :] - mean[1, :]) ** 2))

        return loss



class GaussMixtureCombined(nn.Module):
    # first enropy verion I tried
    __name__ = 'GaussMixtureCombined'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.num_classes = param['num_classes']
        self.epsilon = torch.finfo(torch.float32).eps
        self.two_times_pi = 6.28318530718

    def forward(self, y_pr, y_gt, sample):

        ft = y_pr[1]
        mean = y_pr[2]
        var = y_pr[3]
        class_prob = y_pr[4]

        [n, num_features] = ft.size()
        var = var + self.epsilon

        y_gt = torch.flatten(y_gt).long()

        total_loss = 0.

        for cl in range(self.num_classes):
            current_class = (y_gt == cl)
            ft_cl = ft[current_class, :]
            if ft_cl.size()[0] == 0:
                loss_cl = 0.
            else:
                mean_cl = mean[cl, :]
                var_cl = var[cl, :]

                loss_cl = compute_neg_log_lk(ft_cl, mean_cl, var_cl)
                loss_cl = loss_cl - torch.log(class_prob[cl])

            #####
                out = torch.zeros([ft_cl.size()[0], self.num_classes]).to(self.device)

                for cl_other in range(self.num_classes):
                    mean_cl = mean[cl_other, :]
                    var_cl = var[cl_other, :]
                    next = (ft_cl - mean_cl) ** 2
                    next = next / (2 * var_cl)

                    sigmas_cl = torch.sqrt(var_cl * self.two_times_pi)
                    inside_exp = torch.log(sigmas_cl)

                    next = next + inside_exp

                    next = -next

                out[:, cl_other] = torch.sum(next, dim=1)

                out = out + torch.log(class_prob.to(self.device))
                max_val, _ = torch.max(out, dim=1, keepdim=True)

                out = out - max_val
                out = torch.exp(out)
                out = torch.sum(out, dim=1)
                out = torch.log(out + self.epsilon)
                max_val = torch.squeeze(max_val, dim=1)
                out = out + max_val


                total_loss = total_loss + torch.sum(loss_cl + out)

        return total_loss/n

