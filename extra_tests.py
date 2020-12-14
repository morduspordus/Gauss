
def test_with_matrix(args, model_load, imageSaver=None, EvaluatorIn=None, ft_matrix=None):

    training_type = 'with_blah'
    model = get_model(args)

    if model_load is not None:
        model.load_state_dict(torch.load(model_load))

    num_classes, num_features = ft_matrix.size()
    num_features = num_features - 1

    to_replace = ft_matrix[:, num_features:].flatten()
    model.conv_layer.bias[:] = to_replace

    to_replace = ft_matrix[:, 0:num_features]
    to_replace = to_replace.unsqueeze(dim=2)
    to_replace = to_replace.unsqueeze(dim=2)

    model.conv_layer.weight.data = to_replace

    output_dir = './run/experiments/models'

    dataset_name = "OxfordPet"
    model_name = "yuri"
    im_size = 128
    add_to_file_path, model_save = create_file_name(dataset_name, model_name, im_size, training_type, output_dir)

    logs = test_with_loaded_model(args, model, imageSaver, EvaluatorIn)
    torch.save(model.state_dict(), model_save)

    return logs


def my_own_test(ft_matrix, args, evaluatorIn, model_load):
    valid_dataset = get_val_dataset(args, args['split'])
    dataloader = torch.utils.data.DataLoader(valid_dataset, args['val_batch_size'], shuffle=False,
                                               num_workers=args['num_workers'])

    verbose = args['verbose']
    device = args['device']

    evaluator = evaluatorIn(args)
    evaluator.reset()
    ft_matrix = ft_matrix.to(device)
    ft_matrix = torch.transpose(ft_matrix, 0, 1)
    model = get_model(args)

    if model_load is not None:
        model.load_state_dict(torch.load(model_load))

    model.to(device)
    model.eval()


    with tqdm(dataloader, desc='test', file=sys.stdout, disable=not (verbose)) as iterator:
        for sample in iterator:
            x = sample['image']
            y = sample['label']

            x, y = x.to(device), y.to(device)
            out = model.forward(x)

            if type(out) == tuple:
                ft = out[1]
            else:
                ft = out

            [n, d, h, w] = list(ft.size())
            ft = torch.cat([ft, torch.ones(n, 1, h, w).to(device)], 1)
            #ft = torch.nn.functional.normalize(ft, dim=1, p=2)

            ft = torch.transpose(ft, 0, 1)
            ft = torch.flatten(ft, start_dim=1)
            ft = torch.transpose(ft, 0, 1)

            res = torch.matmul(ft, ft_matrix)
            #res = torch.matmul(ft, torch.transpose(ft_matrix,0,1))
            _, pred = torch.max(res, dim=1)

            pred = pred.view(n, h, w)

            evaluator.add_batch(y, pred)

    results = evaluator.compute_all_metrics()
    print(results)


def compute_means_and_test(model_name, dataset_name, im_size, model_load):

    args = get_standard_arguments(model_name, dataset_name, im_size)

    args['num_features'] = 1539
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True

    args['train_batch_size'] = 8
    args['val_batch_size'] = 8

    evaluator = EvaluatorComputeMean

    _, args['loss_names'] = cross_entropy_loss(ignore_class=255)
    args['split'] = 'train'
    test_logs = T.test(args, model_load, EvaluatorIn=evaluator)

    ft_matrix = test_logs['metrics']['feature_sum']
    print(ft_matrix)
    print(ft_matrix[:, 1539])

    images_dir = './run/Temp/images'
    #imgSaver = ImageSaver(args, images_dir, out_multiplier=1, decode_segmap=decode_segmap_pascal,
    #                      with_gt=True, with_orig_im=True, with_orig_size=True)

    args['split'] = 'val'
    imgSaver = None
    # test_logs = test_with_matrix(args, model_load, imageSaver=imgSaver, ft_matrix=ft_matrix)
    # my_own_test(ft_matrix, args, evaluatorIn=Evaluator, model_load=model_load)
    # #
    # # Do gauss test
    epsilon = 0.000001
    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance'] + epsilon
    class_sizes = ft_matrix[:, args['num_features']]
    sum = torch.sum(class_sizes)
    class_pr = class_sizes/sum

    ft_matrix = mean/var
    mean_by_var = -mean/(2*var)
    mean_t = torch.transpose(mean, 0, 1)
    bias = torch.matmul(mean_by_var, mean_t)
    diag = torch.diagonal(bias)

    # nll = -torch.log(class_pr)
    # diag = diag + nll

    diag = torch.unsqueeze(diag, dim=1)
    ft_matrix = torch.cat((ft_matrix, diag), dim=1)
    test_logs = test_with_matrix(args, model_load, imageSaver=imgSaver, ft_matrix=ft_matrix)
    #my_own_test(ft_matrix, args, evaluatorIn=Evaluator, model_load=model_load)


def temp_test():
    from models.Unet.unet_fixed_features import compute_log_lk

    #r = range(24)
    r = torch.rand([24])

    ft = torch.tensor(r).float()
    ft = ft.view(6, 4).cuda()
    num_classes = 3

    print(ft)

    gt = torch.tensor([0, 0, 1, 1, 2, 2]).cuda()

    model_name = model_names[2]
    dataset_name = singleclass_dataset_names[0]
    im_size = 128
    args = get_standard_arguments(model_name, dataset_name, im_size)
    args['num_features'] = 4
    args['num_classes'] = num_classes
    args['mean_requires_grad'] = False

    evaluator = EvaluatorComputeMean(args)
    evaluator.add_batch(gt, (ft, ft))
    all_metrics = evaluator.compute_all_metrics()
    mean = all_metrics['mean']
    var = all_metrics['variance']

    loss = 0

    [n, c] = ft.size()
    for cl in range(num_classes):
        current_class = (gt == cl)
        ft_cl = ft[current_class, :]
        if ft_cl.size()[0] == 0:
            loss_cl = 0
        else:
            mean_cl = mean[cl, :]
            var_cl = var[cl, :]

            loss_cl = torch.sum(compute_log_lk(ft_cl, mean_cl, var_cl))

        loss = loss + loss_cl

    loss = loss/n
    print("Loss is", loss)

    print("Var is ", var)

    for cl in range(num_classes):
            mean_cl = mean[cl, :]
            var_cl = var[cl, :]

            loss_cl = compute_log_lk(ft, mean_cl, var_cl)

            if cl == 0:
                res = loss_cl[:, None]
            else:
                res = torch.cat((res, loss_cl[:, None]), dim=1)

    print(res)
    # n = 1
    # h = 2
    # w = 3
    # res = res.view(n, h, w, num_classes)
    # print(res)
    # res = torch.transpose(res, 1, 3)
    # res = torch.transpose(res, 2, 3)
    # print(res)
    # print(res.size())
    #
    # gt = torch.tensor([[ [0, 0, 1], [1, 2, 2] ]]).cuda()
    # print(gt.size())
    #
    # evaluator = Evaluator(args)
    # evaluator.add_batch(gt, -res)
    # all_metrics = evaluator.compute_all_metrics()
    # print(all_metrics)

