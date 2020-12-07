import torch
import numpy as np


def compute_edge_mask(image, sigma, adaptive_sigma=False):

    top = image[:, :, :-1, :]
    bottom = image[:, :, 1:, :]
    left = image[:, :, :, :-1]
    right = image[:, :, :, 1:]


    if adaptive_sigma == False:
        mask_h = torch.exp(-1 * torch.sum((left - right) ** 2, dim=1)/ (2 * sigma ** 2))
        mask_v = torch.exp(-1 * torch.sum((top - bottom) ** 2, dim=1) / (2 * sigma ** 2))
    else:
        aver_h = torch.mean(torch.mean(torch.abs(left - right), dim=3), dim=2)
        aver_h = torch.unsqueeze(aver_h, dim=2)
        aver_h = torch.unsqueeze(aver_h, dim=3)

        aver_v = torch.mean(torch.mean(torch.abs(top - bottom), dim=3), dim=2)
        aver_v = torch.unsqueeze(aver_v, dim=2)
        aver_v = torch.unsqueeze(aver_v, dim=3)

        mask_h = torch.exp(-1 * ((left - right) ** 2) / (2 * aver_h ** 2))
        mask_v = torch.exp(-1 * ((top - bottom) ** 2) / (2 * aver_v ** 2))

    return mask_h, mask_v


##----------------------------------------------------------------------------------------------------------------##

def compute_edge_mask_diag(image, sigma):

    image_dims = list(image.size())

    if image_dims[1] > 1:
        image = torch.sum(image,dim=1)

    left_ = image[:, :-1, :]
    left_ = left_[:,:,1:]
    diag1 = image[:, 1:, :]
    diag1 = diag1[:,:, :-1]

    top_ = image[:, :, :-1]
    top_ = top_[:,:-1,:]
    diag2 = image[:, :, 1:]
    diag2 = diag2[:,1:,:]

    mask_d1 = torch.exp(-1 * (left_ - diag1) ** 2 / (2 * sigma ** 2))*0.707
    mask_d2 = torch.exp(-1 * (top_ - diag2) ** 2 / (2 * sigma ** 2))*0.707

    return mask_d1, mask_d2


##----------------------------------------------------------------------------------------------------------------##

def regularized_loss_per_channel_diag(mask_d1, mask_d2, cl, prediction, true_class, negative_class=False):

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

        if prediction is None:
            return 0.
        else:
            mask_d1 = extract_needed_mask(mask_d1, true_class, cl, extract_condition_equal_fn)
            mask_d2 = extract_needed_mask(mask_d2, true_class, cl, extract_condition_equal_fn)

    left = prediction[:,cl ,:-1,:]
    left = left[:,:,1:]
    diag1 = prediction[:, cl , 1:, :]
    diag1 = diag1[:, :, :-1]

    top = prediction[:,cl ,:,:-1]
    top = top[:, :-1,:]
    diag2 = prediction[:,cl ,:,1:]
    diag2 = diag2[:,1:,:]


    h = torch.mean(abs(left - diag1) * mask_d1)
    v = torch.mean(abs(top - diag2) * mask_d2)

    return((h + v)/2.0)

##----------------------------------------------------------------------------------------------------------------##

def  regularized_loss_per_channel(mask_h, mask_v, cl, prediction, true_class, negative_class=False):

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

        if prediction is None:
            return 0.
        else:
            mask_h = extract_needed_mask(mask_h, true_class, cl)
            mask_v = extract_needed_mask(mask_v, true_class, cl)

    top = prediction[:, cl, :-1, :]
    bottom = prediction[:, cl, 1:, :]
    left = prediction[:, cl, :, :-1]
    right = prediction[:, cl, :, 1:]

    h = torch.mean(abs(left - right) * mask_h)
    v = torch.mean(abs(top - bottom) * mask_v)

    return((h + v)/2.0)



def middle_sq_loss_per_channel(cl, prediction, true_class, square_w):

    prediction_pos = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

    if prediction_pos is not None:
        height = prediction_pos.size()[2]
        width = prediction_pos.size()[3]

        middleH = height // 2
        middleW = width // 2

        middle_sq = torch.mean(prediction_pos[:, cl, middleH-square_w:middleH+square_w,middleW-square_w:middleW+square_w])

        loss = (middle_sq - 1.0) ** 2
    else:
        loss = 0.

    prediction_neg = extract_needed_predictions(true_class, prediction, cl, extract_condition_not_equal_fn)
    if prediction_neg is not None:
        loss = loss + (torch.mean(prediction_neg[:, cl, :, :])) ** 2

    return loss


##----------------------------------------------------------------------------------------------------------------##


def loose_sq_loss_per_channel(cl, prediction, size, true_class, negative_class):

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

    if prediction is not None:
        all_dims = list(prediction.size())
        height = all_dims[2]
        width = all_dims[3]
        central_crop = prediction[:, cl, height//2 - height//4 : height//2+height//4, width//2 - width//4 : width//2 + width//4]

        VV,_ = torch.max(central_crop, dim=1)
        HH,_ = torch.max(central_crop, dim=2)

        #loss = ( (torch.mean(VV) - 1) ** 2 + (torch.mean(HH) - 1) ** 2 )/2.0
        loss_v = (torch.mean(VV) - 1) ** 2
        loss_h = (torch.mean(HH) - 1) ** 2
        loss = torch.min(loss_v, loss_h)
    else:
        loss = 0.

    return loss


def five_boxes_helper(crop):
    temp = torch.mean(crop, dim=2)
    temp = torch.mean(temp, dim=1)

    return torch.mean(temp)

def five_boxes_per_channel(cl, prediction):

    if prediction is not None:
        all_dims = list(prediction.size())
        height = all_dims[2]
        width = all_dims[3]

        central_crop = prediction[:, cl, height//2 - height//4 : height//2+height//4, width//2 - width//4 : width//2 + width//4]
        top_left = prediction[:, cl, 0:height//2, 0:width//2]
        top_right = prediction[:, cl, 0:height//2, width//2:]
        bottom_left = prediction[:, cl, height // 2:, 0:width // 2]
        bottom_right = prediction[:, cl, height // 2:, width // 2:]

        lossCC = five_boxes_helper(central_crop)
        lossTL = five_boxes_helper(top_left)
        lossTR = five_boxes_helper(top_right)
        lossBL = five_boxes_helper(bottom_left)
        lossBR = five_boxes_helper(bottom_right)


        loss1 = torch.min(lossCC, lossTL)
        loss2 = torch.min(lossBR, lossBL)
        loss = torch.min(loss1, loss2)
        loss = torch.min(loss, lossTR)

        # all = torch.tensor([lossCC, lossTL, lossTR, lossBL, lossBR]).cuda()
        # weights = torch.softmax(-all, dim=0)
        # loss = torch.sum(all*weights)

    else:
        loss = 0.

    return loss

def class_ce_per_channel(ch, y_pr, true_class):
    __name__ = 'class_ce'

    loss = 0

    y_pr_pos = extract_needed_predictions(true_class, y_pr, ch, extract_condition_equal_fn)

    if y_pr_pos is not None:
        y_pr_pos = y_pr_pos[:, ch, :, :]
        y_pr_pos = -torch.log(y_pr_pos+0.0000001)

        # temp = torch.mean(y_pr_pos, dim=2)
        # temp = torch.mean(temp, dim=1)

        temp1, _ = torch.min(y_pr_pos, dim=2)
        temp2, _ = torch.min(y_pr_pos, dim=1)

        loss = loss + (torch.mean(temp1) + torch.mean(temp2))/2

    y_pr_neg = extract_needed_predictions(true_class, y_pr, ch, extract_condition_not_equal_fn)

    if y_pr_neg is not None:
        y_pr_neg = y_pr_neg[:, ch, :, :]
        y_pr_neg = -torch.log(1-y_pr_neg+0.000001)
        temp = torch.mean(y_pr_neg, dim=2)
        temp = torch.mean(temp, dim=1)
        loss = loss + 10*torch.mean(temp)


    return loss


def class_ce_per_channel2(ch, y_pr, true_class):
    __name__ = 'class_ce'

    loss = 0

    y_pr_pos = extract_needed_predictions(true_class, y_pr, ch, extract_condition_equal_fn)


    if y_pr_pos is not None:
        y_pr_pos = -torch.log(y_pr_pos)
        loss = five_boxes_per_channel(ch, y_pr_pos)

    y_pr_neg = extract_needed_predictions(true_class, y_pr, ch, extract_condition_not_equal_fn)

    if y_pr_neg is not None:
        y_pr_neg = y_pr_neg[:, ch, :, :]
        y_pr_neg = -torch.log(1-y_pr_neg+0.0000001)
        temp = torch.mean(y_pr_neg, dim=2)
        temp = torch.mean(temp, dim=1)
        loss = loss + 10*torch.mean(temp)


    return loss


def new_volume_per_channel(ch, y_pr, true_class, weight):
    __name__ = 'new_volume'

    loss = 0

    y_pr_pos = extract_needed_predictions(true_class, y_pr, ch, extract_condition_equal_fn)

    if y_pr_pos is not None:
        y_pr_pos = y_pr_pos[:, ch, :, :]
        aver = torch.mean(y_pr_pos)
        loss = (aver - weight) ** 2

    return loss


def extract_needed_predictions(true_class, y_pr, cl, cond_fn):

    which_class = cond_fn(true_class, cl)

    needed_prediction = y_pr[which_class, :, :, :]
    if needed_prediction.size()[0] == 0:
        needed_prediction = None

    return needed_prediction


def extract_condition_equal_fn(true_class, cl):
    return true_class == cl



def extract_condition_not_equal_fn(true_class, cl):
    return true_class != cl



def extract_needed_mask(mask, true_class,  cl):

    which_class = (true_class == cl)

    needed_mask = mask[which_class, :, :]
    if needed_mask.size()[0] == 0:
        needed_mask = None

    return needed_mask



def regularized_loss_per_channel_semi(mask, ch, sh, prediction, true_class, negative_class=False):  # olga

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, ch, extract_condition_equal_fn)

        if prediction is None:
            return 0.
        else:
            mask = extract_needed_mask(mask, true_class, cl)

    delta_x = sh[0]
    delta_y = sh[1]

    n, c, height, width = prediction.size()

    if delta_y >= 0:
        first = prediction[:, ch, :height - delta_y, delta_x:]
        second = prediction[:, ch, delta_y:, 0:width - delta_x]
    else:
        delta_y = -delta_y
        first = prediction[:, ch, delta_y:, delta_x:]
        second = prediction[:, ch, 0:height - delta_y, 0:width - delta_x]

    loss = torch.mean(torch.abs(first - second) * mask)

    return loss


def compute_edge_mask_semi(image, sigma_xy, sigma, shift):
    delta_x = shift[0]
    delta_y = shift[1]

    n, f, height, width = image.size()

    if delta_y >= 0:
        first = image[:, :, :height - delta_y, delta_x:]
        second = image[:, :, delta_y:, 0:width - delta_x]

    else:
        delta_y = -delta_y
        first = image[:, :, delta_y:, delta_x:]
        second = image[:, :, 0:height - delta_y, 0:width - delta_x]
    #
    # mask = torch.exp(
    #     -torch.sum((first - second) ** 2, dim=1) / (2 * sigma ** 2) - (delta_x + delta_y) / (2 * sigma_xy ** 2))


    mask = torch.exp(-torch.sum((first - second) ** 2, dim=1) / (2 * sigma ** 2)) * np.sqrt(1 / (delta_x ** 2 + delta_y ** 2))
    return mask
