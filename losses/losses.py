import torch
import torch.nn as nn
from . import loss_utils as UF


##----------------------------------------------------------------------------------------------------------------##
#
#    This file contains losses for weak segmentation CNN, optionally with negative class examples
#    Most losses are designed for single class case:
#           background corresponds to channel 0
#           object corresponds to channel 1
#    But most losses should also work for multiclass case, if manage to extend to multiclass case
#    Losses not applicable to multiclass case have a corresponding comment
#    Ignore class (void class) is denoted with 255 in the mask
#    All pixels in negative class samples have to get classified as background
#    Dataloader should label all pixels in the mask of  negative classes as 0 for cross entropy to work correctly
#
##----------------------------------------------------------------------------------------------------------------##



class GaussianLoss(nn.Module):
    __name__ = 'GaussianLoss'

    def __init__(self, param):
        super().__init__()
        self.device = param['device']
        self.mean = param['mean']
        self.var = param['var']


    def forward(self, y_pr, y_gt, sample):

        if type(y_pr) is tuple:
            ft = y_pr[1]
            y_pr = y_pr[0]

        comp = 1

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

            loss = y_pr*one_hot

            to_add = ft ** 2
            to_add = torch.sum(to_add)
            to_add = to_add / (2*self.var)

            loss = -torch.sum(loss)/y_pr.size()[0] + to_add

            return loss

        else:
            n, c, h, w = y_pr.size()
            y_gt = y_gt.long()

            y_gt = y_gt.flatten()
            y_gt_mask = (y_gt >= 0) & (y_gt < c)

            y_gt = y_gt[y_gt_mask].long()

            II = torch.eye(c)
            one_hot = II[y_gt]

            one_hot = one_hot.to(self.device)

            n, c, h, w = ft.size()
            ft = ft.transpose(1, 2)
            ft = ft.transpose(2, 3)
            ft = ft.reshape(n * w * h, c)
            ft = ft[y_gt_mask, :]

            loss = (ft - self.mean) ** 2
            loss = torch.sum(loss, dim=1)
            loss = loss/(2 * self.var)

            loss = loss * one_hot

            loss = torch.sum(loss)
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


##----------------------------------------------------------------------------------------------------------------##

class SparseCRFLoss(nn.Module):
    __name__ = 'SparseCRFLoss'

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.sigma = param['sigma']
        self.subtract_eps = param['subtract_eps']
        self.with_diag = param['with_diag']
        self.num_classes = param['num_classes']
        self.device = param['device']
        self.negative_class = param['negative_class']

    def empty_solution_cost(self):
        return 0.

    def forward(self, y_pr, y_gt, sample):
        img = sample['image'].to(self.device)
        true_class = sample['image_class']

        if 'edges_h' in sample:
            mask_h = sample['edges_h'].to(self.device)
            mask_v = sample['edges_v'].to(self.device)
            mask_h = mask_h[:, :-1, :]
            mask_v = mask_v[:, :, :-1]

        else:
            mask_h, mask_v = UF.compute_edge_mask(img, self.sigma)
            mask_h = mask_h - self.subtract_eps
            mask_v = mask_v - self.subtract_eps

        loss = 0.
        # regularized loss is not applied to the background class

        for ch in range(1, self.num_classes):
            loss = loss + UF.regularized_loss_per_channel(mask_h, mask_v, ch, y_pr, true_class, self.negative_class)

        if self.with_diag:
            mask_d1, mask_d2 = UF.compute_edge_mask_diag(img, self.sigma)
            mask_d1 = mask_d1 - self.subtract_eps
            mask_d2 = mask_d2 - self.subtract_eps
            loss = loss + UF.regularized_loss_per_channel_diag(mask_d1, mask_d2, 1, y_pr, true_class,
                                                               self.negative_class)

        return self.weight * loss


class MiddleSqLoss(nn.Module):
    __name__ = 'MiddleSqLoss'

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.square_w = param['square_w']
        self.num_classes = param['num_classes']

    def empty_solution_cost(self):
        return self.weight

    def forward(self, y_pr, y_gt, sample):
        loss = 0

        true_class = sample['image_class']

        for ch in range(1, self.num_classes):  # do not compute this loss for the background
            loss = loss + UF.middle_sq_loss_per_channel(ch, y_pr, true_class, self.square_w)

        return self.weight * loss


class VolumeBatchLoss(nn.Module):
    """
    Loss function acts on a batch of images
    """
    __name__ = 'VolumeBatchLoss'

    # cl is the channel for which take volume loss
    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.fraction = param['fraction']
        self.cl = param['cl']
        self.negative_class = param['negative_class']

    def empty_solution_cost(self):
        return (self.fraction ** 2) * self.weight

    def forward(self, y_pr, y_gt, sample):

        true_class = sample['image_class']

        if self.negative_class:
            # extract samples whose true class is not 0, i.e. all but negative samples
            y_pr = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_not_equal_fn)
            if y_pr is None:
                return 0.

        samples_volume = torch.mean(y_pr[:, self.cl, :, :], dim=(1, 2))

        loss = self.weight * ((torch.mean(samples_volume) - self.fraction) ** 2)

        return loss


class VolumeMinLoss(nn.Module):
    """
    Loss function that penalizes for volumes that are too small
    """
    __name__ = 'VolumeMinLoss'

    # cl is the channel for which take volume loss
    def __init__(self, param):
        super().__init__()
        self.weight = param['weight_min']
        self.cl = param['cl']
        self.negative_class = param['negative_class']
        self.vol_min = param['vol_min']

    def empty_solution_cost(self):
        return (self.vol_min ** 2) * self.weight

    def forward(self, y_pr, y_gt, sample):

        true_class = sample['image_class']

        if self.negative_class:
            # extract samples whose true class is not 0, i.e. all but negative samples
            y_pr = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_not_equal_fn)
            if y_pr is None:
                return 0.

        samples_volume = torch.mean(y_pr[:, self.cl, :, :], dim=(1, 2))

        frac_min = torch.min(samples_volume, torch.tensor(self.vol_min).cuda())
        loss_min = torch.mean((frac_min - self.vol_min) ** 2) * self.weight

        return loss_min


class VolumeMaxLoss(nn.Module):
    """
    Loss function that penalizes for volumes that are too large
    Implemented for a single channel currently
    """
    __name__ = 'VolumeMaxLoss'

    # cl is the channel for which take volume loss
    def __init__(self, param):
        super().__init__()
        self.weight = param['weight_max']
        self.cl = param['cl']
        self.negative_class = param['negative_class']
        self.vol_max = param['vol_max']

    def empty_solution_cost(self):
        return self.weight * self.vol_max ** 2

    def forward(self, y_pr, y_gt, sample):

        true_class = sample['image_class']

        if self.negative_class:
            # extract samples whose true class is not 0, i.e. all but negative samples
            y_pr = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_not_equal_fn)
            if y_pr is None:
                return 0.

        samples_volume = torch.mean(y_pr[:, self.cl, :, :], dim=(1, 2))

        frac_max = torch.max(samples_volume, torch.tensor(self.vol_max).cuda())
        loss_max = torch.mean((frac_max - self.vol_max) ** 2) * self.weight

        return loss_max


class BorderLoss(nn.Module):
    # is only applied to the background class, i.e channel 0
    __name__ = 'BorderLoss'

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.border_w = param['border_w']
        self.cl = param['cl']

    def empty_solution_cost(self):
        return 0.

    def forward(self, y_pr, y_gt, sample):
        left = torch.mean(y_pr[:, self.cl, 0:self.border_w, :])
        right = torch.mean(y_pr[:, self.cl, -self.border_w:, :])
        top = torch.mean(y_pr[:, self.cl, :, 0:self.border_w])
        bottom = torch.mean(y_pr[:, self.cl, :, -self.border_w:])

        loss = ((left - 1.0) ** 2 + (right - 1.0) ** 2 + (top - 1.0) ** 2 + (bottom - 1.0) ** 2) / 4.0

        return self.weight * loss


class LooseSqLoss(nn.Module):
    # currently implemented only for a single class case
    __name__ = 'LooseSqLoss'

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.size = param['size']
        self.cl = param['cl']
        self.negative_class = param['negative_class']

    def empty_solution_cost(self):
        return self.weight

    def forward(self, y_pr, y_gt, sample):
        true_class = sample['image_class']
        loss = UF.loose_sq_loss_per_channel(self.cl, y_pr, self.size, true_class, self.negative_class)
        return self.weight * loss


class NegativeClassLoss(nn.Module):
    __name__ = 'NegativeClassLoss'

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']

    def empty_solution_cost(self):
        return 0.

    def forward(self, y_pr, y_gt, sample):

        true_class = sample['image_class']
        # extract negative class samples
        negative_samples = UF.extract_needed_predictions(true_class, y_pr, 0, UF.extract_condition_equal_fn)

        if negative_samples is not None:
            loss = torch.mean(negative_samples[:, 1, :, :])  # the channel corresponding to object should be 0
            loss = loss ** 2
        else:
            loss = 0.

        return self.weight * loss


class SemiSparseCRFLoss(nn.Module):
    __name__ = 'SemiSparseCRFLoss'

    def empty_solution_cost(self):
        return 0.

    def __init__(self, param):
        super().__init__()
        self.weight = param['weight']
        self.sigma = param['sigma']
        self.sigma_xy = param['sigma_xy']
        self.subtract_eps = param['subtract_eps']
        self.negative_class = param['negative_class']
        self.num_classes = param['num_classes']
        self.shifts = param['shifts']
        self.device = param['device']

    def forward(self, y_pr, y_gt, sample):

        loss = 0.
        true_class = sample['image_class']
        img = sample['image'].to(self.device)

        for sh in self.shifts:
            mask = UF.compute_edge_mask_semi(img, self.sigma_xy, self.sigma, sh)
            mask = mask - self.subtract_eps

            # regularized loss is not applied to the background class
            for ch in range(1, self.num_classes):
                loss = loss + UF.regularized_loss_per_channel_semi(mask, ch, sh, y_pr, true_class, self.negative_class)

        return self.weight * loss / len(self.shifts)

