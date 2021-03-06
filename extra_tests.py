
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


def visualize_pca():
    model_load = './run/experiments/models/OxfordPet_MobileNetV2_Ft_LinearFixed_128__gauss__V30.pt'
    model_name = model_names[2]
    dataset_name = singleclass_dataset_names[0]
    im_size = 128

    args = get_standard_arguments(model_name, dataset_name, im_size)

    args['num_features'] = 1536
    args['model_load'] = model_load
    args['num_classes'] = 3
    args['cats_dogs_separate'] = True

    args['train_batch_size'] = 4
    args['val_batch_size'] = 8
    args['mean_requires_grad'] = False
    args['mean'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization
    args['var'] = torch.rand(args['num_classes'], args['num_features'])  # value is not important, for inititalization



    evaluator = EvaluatorComputeMeanSomeFeatures

    _, args['loss_names'] = gaussian_loss(args)

    args['split'] = 'train'
    args['shuffle_test'] = True

    test_logs = T.test(args, model_load, EvaluatorIn=evaluator)

    mean = test_logs['metrics']['mean']
    var = test_logs['metrics']['variance']
    ft = test_logs['metrics']['ft']
    gt = test_logs['metrics']['gt']
    ft = torch.transpose(ft, 0, 1)

    ft = ft.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    from sklearn import decomposition

    pca = decomposition.PCA(n_components=3)
    pca.fit(ft)
    principalComponents = pca.transform(ft)
    mean_pca = pca.transform(mean.cpu().detach().numpy())

    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(ft)
    # mean_pca = pca.fit(mean.cpu().detach().numpy())


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0]
    #colors = ['r', 'g', 'b']
    colors = ['r']
    for target, color in zip(targets, colors):
        indicesToKeep = gt == target
        ax.scatter(principalComponents[indicesToKeep, 0],
                   principalComponents[indicesToKeep, 1],
                   principalComponents[indicesToKeep, 2],
                   c=color)
        ax.scatter(mean_pca[0, 0], mean_pca[0, 1], mean_pca[0, 2], c='g')

    ax.legend(targets)
    ax.grid()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [1]
    # colors = ['r', 'g', 'b']
    colors = ['g']
    for target, color in zip(targets, colors):
        indicesToKeep = gt == target
        ax.scatter(principalComponents[indicesToKeep, 0],
                   principalComponents[indicesToKeep, 1],
                   principalComponents[indicesToKeep, 2],
                   c=color)
        ax.scatter(mean_pca[1, 0], mean_pca[1, 1], mean_pca[1, 2], c='r')
    ax.legend(targets)
    ax.grid()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [2]
    # colors = ['r', 'g', 'b']
    colors = ['b']
    for target, color in zip(targets, colors):
        indicesToKeep = gt == target
        ax.scatter(principalComponents[indicesToKeep, 0],
                   principalComponents[indicesToKeep, 1],
                   principalComponents[indicesToKeep, 2],
                   c=color)

        ax.scatter(mean_pca[2, 0], mean_pca[2, 1], mean_pca[2, 2], c='g')
    ax.legend(targets)
    ax.grid()
    plt.show()

    print("done")


def new_test():

    epsilon = torch.finfo(torch.float32).eps

    r = torch.rand([24])

    ft = torch.tensor(r).float()
    ft = ft.view(6, 4)
    num_classes = 3

    # mean = torch.rand([3,4])
    # var = torch.rand([3,4])
    # class_prob = torch.rand([3])
    # class_prob = class_prob/sum(class_prob)

    ft = torch.tensor([[0.0535, 0.7935, 0.2525, 0.3562],
            [0.7587, 0.0454, 0.1192, 0.0193],
            [0.5526, 0.1416, 0.7261, 0.2430],
            [0.1548, 0.0698, 0.1974, 0.7610],
            [0.8434, 0.5806, 0.4745, 0.6355],
            [0.3435, 0.8416, 0.3451, 0.8155]])
    mean = torch.tensor([[0.0213, 0.2804, 0.3500, 0.0705],
            [0.5077, 0.1460, 0.6972, 0.7937],
            [0.6053, 0.5292, 0.9267, 0.7868]])

    var = torch.tensor([[0.7306, 0.2218, 0.3051, 0.1479],
            [0.5749, 0.1398, 0.1326, 0.5127],
            [0.5174, 0.8227, 0.6534, 0.3736]])
    class_prob = torch.tensor([0.0836, 0.0315, 0.8849])

    n = 6

    print(ft)
    print(mean)
    print(var)
    print(class_prob)

    two_times_pi = 6.28318530718

    out = torch.zeros([n, num_classes])

    for cl in range(num_classes):
        mean_cl = mean[cl, :]
        var_cl = var[cl, :]

        next = (ft - mean_cl) ** 2
        next = next / (2 * var_cl)

        sigmas_cl = torch.sqrt(var_cl * two_times_pi)
        inside_exp = torch.log(sigmas_cl)

        next = next + inside_exp

        next = -next

        out[:, cl] = torch.sum(next, dim=1)

    out = out + torch.log(class_prob)
    max_val, _ = torch.max(out, dim=1, keepdim=True)

    out = out - max_val
    out = torch.exp(out)
    out = torch.sum(out, dim=1)
    out = torch.log(out + epsilon)
    max_val = torch.squeeze(max_val, dim=1)
    out = out + max_val
    out = torch.mean(out)
    print(out)

