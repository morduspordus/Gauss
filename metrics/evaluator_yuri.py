import torch
from metrics.evaluator import Evaluator


class EvaluatorComputeMean(Evaluator):

    def __init__(self, args):
        super(EvaluatorComputeMean, self).__init__(args)
        self.num_features = args['num_features']
        self.feature_matrix = torch.zeros((self.num_classes, self.num_features + 1)).to(self.device)
        self.feature_matrix_sq = torch.zeros((self.num_classes, self.num_features + 1)).to(self.device)
        self.ignore_class = args['ignore_class']
        self.num_classes = args['num_classes']
        self.device = args['device']
        assert (self.ignore_class >= self.num_classes or self.ignore_class == -1)

    def _generate_matrix(self, gt, ft):

        if type(ft) is tuple:  # first entry in tuple is classification result, second is features
            ft = ft[1]

        [n, d, h, w] = list(ft.size())
        ft = torch.cat([ft, torch.ones(n, 1, h, w).to(self.device)], 1)
        ft_sq = ft ** 2

        gt = gt.long()

        if self.ignore_class > 0:
            gt[gt == self.ignore_class] = self.num_classes
            num_classes_for_one_hot = self.num_classes + 1
        else:
            num_classes_for_one_hot = self.num_classes

        ft = torch.transpose(ft, 0, 1)
        ft = torch.flatten(ft, start_dim=1)

        ft_sq = torch.transpose(ft_sq, 0, 1)
        ft_sq = torch.flatten(ft_sq, start_dim=1)

        gt = torch.flatten(gt)
        gt = torch.nn.functional.one_hot(gt, num_classes=num_classes_for_one_hot).float()

        out = torch.matmul(ft, gt)
        out = torch.transpose(out, 0, 1)

        out_sq = torch.matmul(ft_sq, gt)
        out_sq = torch.transpose(out_sq, 0, 1)

        size = list(gt.shape)
        if size[1] > self.num_classes:
            out = out[0:-1, :]
            out_sq = out_sq[0:-1, :]
        return out, out_sq

    # second way to generate feature_matrix, equivalent to the first way
    def _generate_matrix_other(self, gt, ft):

        if type(ft) is tuple:
            ft = ft[1]

        [n, d, h, w] = list(ft.size())
        ft = torch.cat([ft, torch.ones(n, 1, h, w).to(self.device)], 1)

        gt = gt.long()

        ft = torch.flatten(ft, start_dim=2)
        ft = ft.transpose(0, 1)
        ft = torch.flatten(ft, start_dim=1)
        ft = ft.transpose(0, 1)

        [n, h, w] = list(gt.size())
        gt = torch.flatten(gt)

        for cl in range(self.num_classes):
            current_class = (gt == cl)
            needed_features = ft[current_class, :]
            needed_features = torch.sum(needed_features, dim=0)
            needed_features = torch.unsqueeze(needed_features, 0)

            if cl == 0:
                all_features = needed_features
            else:
                all_features = torch.cat((all_features, needed_features), dim=0)

        return all_features

    def add_batch(self, gt, ft):
        ft_, ft_sq_ = self._generate_matrix(gt, ft)
        self.feature_matrix += ft_
        self.feature_matrix_sq += ft_sq_

    def reset(self):
        self.feature_matrix = torch.zeros((self.num_classes, self.num_features + 1)).to(self.device)
        self.feature_matrix_sq = torch.zeros((self.num_classes, self.num_features + 1)).to(self.device)


    def mean(self):
        divide_by = self.feature_matrix[:, self.num_features]
        divide_by = torch.unsqueeze(divide_by, dim=1)
        mean_vector = self.feature_matrix[:, 0:self.num_features]/divide_by
        return mean_vector

    def variance(self):
        mean_vector = self.mean()
        mean_vector = mean_vector[:, 0:self.num_features]
        divide_by = self.feature_matrix_sq[:, self.num_features]
        divide_by = torch.unsqueeze(divide_by, dim=1)
        var_vector = self.feature_matrix_sq[:, 0:self.num_features] / divide_by
        var_vector = var_vector - mean_vector ** 2
        return var_vector

    def ft_matrix(self):
        matrix = self.feature_matrix
        #matrix = torch.nn.functional.normalize(matrix, dim=1, p=2)

        return matrix

    def compute_all_metrics(self):
        all_metrics = {}

        all_metrics['feature_matrix'] = self.ft_matrix()
        all_metrics['mean'] = self.mean()
        all_metrics['variance'] = self.variance()
        all_metrics['feature_matrix_sq'] = self.feature_matrix_sq

        return all_metrics
