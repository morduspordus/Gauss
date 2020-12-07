
def extra_test():
    model_name = model_names[15]
    dataset_name = singleclass_dataset_names[5]
    im_size = 128
    args = get_standard_arguments(model_name, dataset_name, im_size)
    args['num_features'] = 5
    evaluator = EvaluatorComputeMean(args)

    device = args['device']
    ft = torch.tensor([[1, 2, 3, 4, 1], [5, 6, 7, 8, 0], [6, 5, 8, 7, 0],  [9, 10, 11, 12, 1], [13, 14, 15, 16, 0], [14, 13, 16, 15, 0]]).to(device)
    ft = torch.transpose(ft, 0, 1)
    ft = torch.unsqueeze(ft, dim=0)
    ft = torch.unsqueeze(ft, dim=3)
    gt = torch.tensor([0, 0, 0, 1, 1, 1]).to(device)
    gt = torch.unsqueeze(gt, dim=0)
    gt = torch.unsqueeze(gt, dim=2)

    # print(ft)
    #print(ft.size())
    # print(gt)
    #print(gt.size())

    evaluator.add_batch(gt, ft)
    all_metrics = evaluator.compute_all_metrics()
    #print(all_metrics)

    ft_matrix = all_metrics['feature_matrix']
    print("Feature matrix ", ft_matrix)

    [n, d, h, w] = list(ft.size())
    ft = torch.cat([ft, torch.ones(n, 1, h, w).to(device)], 1)
    # ft = torch.nn.functional.normalize(ft, dim=1, p=2)

    ft = torch.transpose(ft, 0, 1)
    ft = torch.flatten(ft, start_dim=1)
    ft = torch.transpose(ft, 0, 1)

    print('features\n', ft)
    print('ft_matrix\n', ft_matrix)
    res = torch.matmul(ft, torch.transpose(ft_matrix, 0, 1))
    r, pred = torch.max(res, dim=1)

    pred = pred.view(n, h, w)
    print(pred)
    print(res)

    # Do gauss test
    # epsilon = 0.000001
    # mean = all_metrics['mean']
    # var = all_metrics['variance'] + epsilon
    # ft_matrix = mean/var
    # mean_by_var = -mean/(2*var)
    # mean_t = torch.transpose(mean, 0, 1)
    # bias = torch.matmul(mean_by_var, mean_t)
    # diag = torch.diagonal(bias)
    # diag = torch.unsqueeze(diag, dim=1)
    # ft_matrix = torch.cat((ft_matrix, diag), dim=1)
    #
    # [n, d, h, w] = list(ft.size())
    # ft = torch.cat([ft, torch.ones(n, 1, h, w).to(device)], 1)
    # # ft = torch.nn.functional.normalize(ft, dim=1, p=2)
    #
    # ft = torch.transpose(ft, 0, 1)
    # ft = torch.flatten(ft, start_dim=1)
    # ft = torch.transpose(ft, 0, 1)
    #
    # print('features\n', ft)
    # print('ft_matrix\n', ft_matrix)
    # res = torch.matmul(ft, torch.transpose(ft_matrix, 0, 1))
    # r, pred = torch.max(res, dim=1)
    #
    # pred = pred.view(n, h, w)
    # print(pred)
    # print(res)

